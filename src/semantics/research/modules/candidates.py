from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

import nltk
import torch
from huggingface_hub import snapshot_download
from sentence_transformers import CrossEncoder, SentenceTransformer, util

if TYPE_CHECKING:
    from ..config import CandidatesConfig as PydanticCandidatesConfig

from .utils.logging import info_print

logger = logging.getLogger(__name__)

# --- 0. Setup NLTK for sentence splitting ---
for _resource in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{_resource}")
    except LookupError:
        logger.debug("NLTK '%s' tokenizer not found. Downloading...", _resource)
        nltk.download(_resource, quiet=True)

# --- Configuration Class ---
@dataclass
class RetrievalConfig:
    """Configuration for the retrieval system."""
    retriever_model: str = 'nomic-ai/nomic-embed-text-v1.5'
    reranker_model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    max_chunk_tokens: int = 128
    overlap_tokens: int = 32
    retrieval_k: int = 50
    rerank_batch_size: int = 32
    aggregation_method: Literal['max', 'mean', 'top_k_mean', 'weighted_mean'] = 'max'
    # For filtering: only return items with scores above this threshold
    min_score_threshold: float = -5.0  # Cross-encoder scores can be negative
    # For filtering: minimum score difference from top result (helps filter irrelevant results)
    score_margin_threshold: float = None  # e.g., 2.0 means only keep items within 2 points of best
    verbose: bool = True

# --- 1. Load Models ---
def _resolve_retriever_model(name: str) -> str:
    """Normalize short-hand model identifiers to fully-qualified repos."""

    if not name:
        return name

    if "/" in name:
        return name

    if name.startswith("nomic-"):
        return f"nomic-ai/{name}"

    return name


def load_models(config: RetrievalConfig):
    """Load retriever and reranker models."""
    retriever_model_id = _resolve_retriever_model(config.retriever_model)

    if retriever_model_id.startswith("nomic-ai/"):
        try:
            import einops  # noqa: F401  # ensure dependency present for dynamic modules
        except ImportError as exc:  # pragma: no cover - indicates misconfigured environment
            raise RuntimeError(
                "The nomic retriever requires the 'einops' package. "
                "Install it via pip within the research environment."
            ) from exc

    if not config.verbose:
        logging.getLogger("transformers_modules").setLevel(logging.ERROR)
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        logging.getLogger("sentence_transformers.SentenceTransformer").setLevel(logging.ERROR)

    cache_root = Path(os.environ.get("SENTENCE_TRANSFORMERS_HOME", Path.home() / ".cache" / "sentence-transformers"))
    cache_root.mkdir(parents=True, exist_ok=True)

    local_override = os.environ.get("NOMIC_EMBED_MODEL_DIR")
    retriever_source = local_override or retriever_model_id

    if config.verbose:
        logger.debug("Loading retriever model (%s)...", retriever_model_id)
    try:
        retriever = SentenceTransformer(
            retriever_source,
            trust_remote_code=True,
            cache_folder=str(cache_root),
        )
    except (OSError, ValueError) as exc:
        if local_override:
            raise RuntimeError(
                f"Failed to load retriever model from override directory: {local_override}"
            ) from exc

        try:
            local_snapshot = snapshot_download(repo_id=retriever_model_id, token=None)
            retriever = SentenceTransformer(local_snapshot, trust_remote_code=True)
        except Exception as download_exc:
            raise RuntimeError(
                "Unable to download the retriever model without a Hugging Face token. "
                "Provide a local checkout via NOMIC_EMBED_MODEL_DIR or ensure anonymous access is allowed."
            ) from download_exc

    if config.verbose:
        logger.debug("Retriever model loaded.")
    
    if config.verbose:
        logger.debug("Loading reranker model (%s)...", config.reranker_model)
    reranker = CrossEncoder(config.reranker_model)
    if config.verbose:
        logger.debug("Reranker model loaded.")
    
    return retriever, reranker

def load_data(json_file_path: str) -> List[Dict[str, Any]]:
    """Loads the list of items from a JSON file."""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                logger.warning("Expected a list in %s, got %s", json_file_path, type(data))
                return []
            return data
    except FileNotFoundError:
        logger.warning("File not found: %s", json_file_path)
        return []
    except json.JSONDecodeError as e:
        logger.warning("Invalid JSON in %s: %s", json_file_path, e)
        return []
    except Exception as e:
        logger.warning("Error loading %s: %s", json_file_path, e)
        return []

# --- 2. Improved Chunking Function ---
def create_corpus_chunks(items_list: List[Dict[str, Any]], config: RetrievalConfig) -> List[Dict[str, Any]]:
    """
    Splits items into chunks based on token count with smart sentence boundaries.
    Uses semantic-aware chunking that respects sentence boundaries.
    """
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        use_tokenizer = True
    except Exception as e:
        logger.debug("Could not load tokenizer (%s). Falling back to sentence-based chunking.", e)
        use_tokenizer = False
    
    if not items_list:
        logger.debug("Empty items list provided for chunking.")
        return []
    
    if config.verbose:
        logger.debug("Creating chunks from %d items...", len(items_list))
    chunk_database = []
    
    for i, item in enumerate(items_list):
        if not isinstance(item, dict):
            logger.debug("Item at index %d is not a dictionary. Skipping.", i)
            continue
            
        title = item.get('title', '')
        text = item.get('text', '')
        
        if not text or not text.strip():
            if config.verbose:
                logger.debug("Item at index %d has no text. Skipping.", i)
            continue
        
        try:
            sentences = nltk.sent_tokenize(text)
        except Exception as e:
            logger.debug("Error tokenizing text for item %d: %s", i, e)
            continue
        
        if not sentences:
            continue
        
        if use_tokenizer:
            # Token-based chunking
            current_chunk = []
            current_tokens = 0
            
            for sentence in sentences:
                try:
                    sentence_tokens = len(tokenizer.encode(sentence, add_special_tokens=False))
                except Exception as e:
                    logger.debug("Error encoding sentence: %s. Skipping.", e)
                    continue
                
                # If adding this sentence exceeds max tokens and we have content, save chunk
                if current_tokens + sentence_tokens > config.max_chunk_tokens and current_chunk:
                    chunk_text = " ".join(current_chunk)
                    contextual_chunk = f"Title: {title}. Text: {chunk_text}"
                    chunk_database.append({
                        'parent_item_index': i,
                        'chunk_text': contextual_chunk
                    })
                    
                    # Start new chunk with overlap (keep last sentence if within overlap limit)
                    if config.overlap_tokens > 0 and len(current_chunk) > 1:
                        last_sentence = current_chunk[-1]
                        last_sentence_tokens = len(tokenizer.encode(last_sentence, add_special_tokens=False))
                        if last_sentence_tokens <= config.overlap_tokens:
                            current_chunk = [last_sentence]
                            current_tokens = last_sentence_tokens
                        else:
                            current_chunk = []
                            current_tokens = 0
                    else:
                        current_chunk = []
                        current_tokens = 0
                
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
            
            # Add final chunk if it has content
            if current_chunk:
                chunk_text = " ".join(current_chunk)
                contextual_chunk = f"Title: {title}. Text: {chunk_text}"
                chunk_database.append({
                    'parent_item_index': i,
                    'chunk_text': contextual_chunk
                })
        else:
            # Fallback: sentence-based chunking
            sentences_per_chunk = 3
            overlap = 1
            step = sentences_per_chunk - overlap
            
            for j in range(0, len(sentences), step):
                chunk_sentences = sentences[j : j + sentences_per_chunk]
                chunk_text = " ".join(chunk_sentences)
                contextual_chunk = f"Title: {title}. Text: {chunk_text}"
                
                chunk_database.append({
                    'parent_item_index': i,
                    'chunk_text': contextual_chunk
                })
                
                if j + sentences_per_chunk >= len(sentences):
                    break
    
    if config.verbose:
        logger.debug("Created %d chunks.", len(chunk_database))
    return chunk_database

# --- 3. Embedding Function ---
def create_chunk_embeddings(
    chunk_database: List[Dict[str, Any]],
    retriever_model: SentenceTransformer,
    *,
    verbose: bool = False,
) -> torch.Tensor:
    """Creates embeddings for all text chunks."""
    if not chunk_database:
        logger.debug("Empty chunk database provided.")
        return None
    
    if verbose:
        logger.debug("Computing embeddings for %d chunks...", len(chunk_database))
    start_time = time.time()
    
    try:
        corpus_texts = [chunk['chunk_text'] for chunk in chunk_database]
        
        # Nomic's required prefix for documents
        prefixed_corpus = ["search_document: " + text for text in corpus_texts]
    
        corpus_embeddings = retriever_model.encode(
            prefixed_corpus, 
            convert_to_tensor=True, 
            show_progress_bar=verbose,
            batch_size=32  # Add batch size for efficiency
        )
        
        end_time = time.time()
        if verbose:
            logger.debug("Embeddings computed in %.2f seconds.", end_time - start_time)
        return corpus_embeddings
    except Exception as e:
        logger.warning("Error creating embeddings: %s", e)
        return None

# --- 4. Aggregation Strategy ---
def aggregate_chunk_scores(chunk_scores: List[float], method: str = 'max') -> float:
    """
    Aggregate multiple chunk scores for the same item.
    
    For reliability in filtering relevant vs irrelevant content:
    - 'max': Best for precision - if ANY chunk is highly relevant, the item is relevant
    - 'top_k_mean': More conservative - requires multiple good chunks
    - 'mean': Most conservative - all chunks must be reasonably relevant
    - 'weighted_mean': Emphasizes best chunks but considers all
    
    Args:
        chunk_scores: list of reranker scores for chunks from the same item
        method: aggregation strategy
    """
    if not chunk_scores:
        return float('-inf')
    
    if method == 'max':
        # Best single chunk represents the item - good for finding needle in haystack
        return max(chunk_scores)
    
    elif method == 'mean':
        # Average of all chunks - penalizes items with mixed relevance
        return sum(chunk_scores) / len(chunk_scores)
    
    elif method == 'top_k_mean':
        # Average of top 3 chunks (or fewer if not available)
        # Balances between max and mean - reduces noise from single outlier chunk
        k = min(3, len(chunk_scores))
        top_scores = sorted(chunk_scores, reverse=True)[:k]
        return sum(top_scores) / len(top_scores)
    
    elif method == 'weighted_mean':
        # Exponentially weight higher scores more heavily
        # Good middle ground - emphasizes best chunks but not as extreme as max
        weights = [math.exp(s) for s in chunk_scores]
        total_weight = sum(weights)
        if total_weight == 0:
            return sum(chunk_scores) / len(chunk_scores)
        return sum(s * w for s, w in zip(chunk_scores, weights)) / total_weight
    
    else:
        logger.debug("Unknown aggregation method '%s'. Using 'max'.", method)
        return max(chunk_scores)

# --- 5. Ranking Function with All Improvements ---
def rank_items(
    query: str,
    original_items: List[Dict[str, Any]],
    chunk_database: List[Dict[str, Any]],
    corpus_embeddings: torch.Tensor,
    retriever_model: SentenceTransformer,
    reranker_model: CrossEncoder,
    config: RetrievalConfig,
    final_top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Ranks original items based on reranked scores of their chunks.
    Includes comprehensive error handling, diagnostics, and filtering.
    """
    
    # === INPUT VALIDATION ===
    if not query or not query.strip():
        logger.debug("Empty query provided.")
        return []
    
    if not original_items:
        logger.debug("No items provided.")
        return []
    
    if not chunk_database:
        logger.debug("No chunks available.")
        return []
    
    if corpus_embeddings is None:
        logger.debug("Corpus embeddings are not available.")
        return []
    
    if len(corpus_embeddings) != len(chunk_database):
        logger.warning("Embedding count (%d) doesn't match chunk count (%d)", len(corpus_embeddings), len(chunk_database))
        return []
    
    query = query.strip()
    if config.verbose:
        logger.debug("SEARCHING FOR: '%s'", query)
    
    # === STAGE 1: RETRIEVAL (Bi-Encoder) ===
    retrieval_k = min(config.retrieval_k, len(chunk_database))
    
    try:
        # Nomic's required prefix for queries
        query_with_prefix = "search_query: " + query
        query_embedding = retriever_model.encode(query_with_prefix, convert_to_tensor=True)
        
        # Find the top 'retrieval_k' chunks
        retrieval_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_retrieval_results = torch.topk(retrieval_scores, k=retrieval_k)
        
        retrieved_hits = []
        for score, idx in zip(top_retrieval_results[0], top_retrieval_results[1]):
            retrieved_hits.append({
                'chunk_index': idx.item(),
                'retrieval_score': score.item()
            })
        
        if config.verbose:
            logger.debug("[STAGE 1: RETRIEVAL] Retrieved %d candidate chunks", len(retrieved_hits))
            unique_items = len(set(chunk_database[h['chunk_index']]['parent_item_index'] for h in retrieved_hits))
            logger.debug("  Representing %d unique items", unique_items)
            logger.debug("  Retrieval score range: [%.3f, %.3f]",
                         min(h['retrieval_score'] for h in retrieved_hits),
                         max(h['retrieval_score'] for h in retrieved_hits))
    
    except Exception as e:
        logger.warning("Error during retrieval stage: %s", e)
        return []
    
    # === STAGE 2: RERANKING (Cross-Encoder) ===
    try:
        # Create pairs of (query, chunk_text) for the reranker
        rerank_input = []
        for hit in retrieved_hits:
            chunk_text = chunk_database[hit['chunk_index']]['chunk_text']
            rerank_input.append((query, chunk_text))
        
        # Process in batches to avoid memory issues
        cross_scores = []
        batch_size = config.rerank_batch_size
        
        if config.verbose:
            logger.debug("[STAGE 2: RERANKING] Processing %d pairs in batches of %d",
                         len(rerank_input), batch_size)
        
        for i in range(0, len(rerank_input), batch_size):
            batch = rerank_input[i:i + batch_size]
            batch_scores = reranker_model.predict(batch)
            cross_scores.extend(batch_scores)
        
        # Add rerank scores and parent item info to our hits
        for i, score in enumerate(cross_scores):
            hit = retrieved_hits[i]
            hit['rerank_score'] = float(score)
            chunk_index = hit['chunk_index']
            hit['parent_item_index'] = chunk_database[chunk_index]['parent_item_index']
        
        if config.verbose:
            rerank_scores_list = [h['rerank_score'] for h in retrieved_hits]
            logger.debug("  Rerank score range: [%.3f, %.3f]",
                         min(rerank_scores_list), max(rerank_scores_list))
            logger.debug("  Mean rerank score: %.3f",
                         sum(rerank_scores_list) / len(rerank_scores_list))
    
    except Exception as e:
        logger.warning("Error during reranking stage: %s", e)
        return []
    
    # === STAGE 3: AGGREGATION ===
    try:
        if config.verbose:
            logger.debug("[STAGE 3: AGGREGATION] method=%s", config.aggregation_method)
        
        # Group chunks by parent item
        item_chunk_scores = {}
        item_chunk_details = {}  # For diagnostics
        
        for hit in retrieved_hits:
            parent_index = hit['parent_item_index']
            rerank_score = hit['rerank_score']
            
            if parent_index not in item_chunk_scores:
                item_chunk_scores[parent_index] = []
                item_chunk_details[parent_index] = []
            
            item_chunk_scores[parent_index].append(rerank_score)
            item_chunk_details[parent_index].append({
                'chunk_index': hit['chunk_index'],
                'score': rerank_score
            })
        
        # Aggregate scores for each item
        final_item_scores = {}
        for parent_index, scores in item_chunk_scores.items():
            final_item_scores[parent_index] = aggregate_chunk_scores(
                scores, 
                method=config.aggregation_method
            )
        
        if config.verbose:
            logger.debug("  Aggregated scores for %d unique items", len(final_item_scores))
        
    except Exception as e:
        logger.warning("Error during aggregation stage: %s", e)
        return []
    
    # === STAGE 4: FILTERING & RANKING ===
    try:
        # Sort items by aggregated score
        sorted_item_indices = sorted(
            final_item_scores.keys(),
            key=lambda x: final_item_scores[x],
            reverse=True
        )
        
        # Apply score thresholding for quality filtering
        filtered_items = []
        best_score = final_item_scores[sorted_item_indices[0]] if sorted_item_indices else float('-inf')
        
        if config.verbose:
            logger.debug("[STAGE 4: FILTERING & RANKING]")
            logger.debug("  Best score: %.3f", best_score)
            logger.debug("  Min score threshold: %s", config.min_score_threshold)
            if config.score_margin_threshold:
                logger.debug("  Score margin threshold: %s", config.score_margin_threshold)
        
        for item_index in sorted_item_indices:
            score = final_item_scores[item_index]
            
            # Filter by absolute threshold
            if score < config.min_score_threshold:
                if config.verbose:
                    logger.debug("  Filtered out item %d (score %.3f below threshold)", item_index, score)
                continue
            
            # Filter by margin from best score
            if config.score_margin_threshold is not None:
                score_diff = best_score - score
                if score_diff > config.score_margin_threshold:
                    if config.verbose:
                        logger.debug("  Filtered out item %d (score %.3f, %.3f below best)", item_index, score, score_diff)
                    continue
            
            filtered_items.append(item_index)
        
        # Limit to top K
        final_items = filtered_items[:final_top_k]
        
        if config.verbose:
            logger.debug("  Returned top %d items (requested: %d)", len(final_items), final_top_k)
        
    except Exception as e:
        logger.warning("Error during filtering stage: %s", e)
        return []
    
    # === FINAL RESULTS ===
    if config.verbose:
        logger.debug("TOP %d RESULTS", len(final_items))
    
    ranked_list = []
    for rank, item_index in enumerate(final_items, 1):
        try:
            original_item = original_items[item_index]
            best_score = final_item_scores[item_index]
            num_chunks = len(item_chunk_scores[item_index])
            chunk_scores = item_chunk_scores[item_index]

            if config.verbose:
                logger.debug("%d. [%.3f] %s", rank, best_score, original_item.get('title', 'Untitled'))
                logger.debug("   Item Index: %d", item_index)
                logger.debug("   Chunks: %d | Top scores: %s", num_chunks,
                             [f'{s:.2f}' for s in sorted(chunk_scores, reverse=True)[:3]])

                text_preview = original_item.get('text', '')[:200]
                if len(original_item.get('text', '')) > 200:
                    text_preview += "..."
                logger.debug("   Preview: %s", text_preview)

            ranked_list.append({
                "rank": rank,
                "item": original_item,
                "aggregated_score": best_score,
                "num_chunks": num_chunks,
                "chunk_scores": chunk_scores,
                "item_index": item_index
            })

        except Exception as e:
            logger.debug("Error processing item at index %d: %s", item_index, e)
            continue
    
    return ranked_list

# --- Convenience entry point ---
def select_candidates_from_search_results(
    query: str,
    search_payload: Union[Dict[str, Any], List[Dict[str, Any]]],
    *,
    config: Optional[RetrievalConfig] = None,
    final_top_k: int = 25,
    save_path: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """Run the retrieval pipeline against a search payload and return ranked candidates."""

    if not query or not query.strip():
        logger.debug("Empty query provided for candidate selection.")
        return []

    config = config or RetrievalConfig()

    info_print("Selecting best candidates")

    if isinstance(search_payload, dict):
        raw_results = search_payload.get("results", [])
    else:
        raw_results = search_payload

    if not raw_results:
        logger.debug("No search results provided for candidate selection.")
        return []

    if config.verbose:
        logger.debug("Preparing %d search results for candidate selection...", len(raw_results))

    normalized_items: List[Dict[str, Any]] = []
    for idx, result in enumerate(raw_results):
        if not isinstance(result, dict):
            if config.verbose:
                logger.debug("Search result at index %d is not a dictionary. Skipping.", idx)
            continue

        title_raw = result.get("title") or result.get("url") or f"Result {idx + 1}"
        title = title_raw.strip() if isinstance(title_raw, str) else str(title_raw)

        description_candidate = result.get("description") or result.get("snippet")
        description = description_candidate.strip() if isinstance(description_candidate, str) else ""

        additional_fragments: List[str] = []
        for key in ("summary", "text"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                additional_fragments.append(value.strip())

        if result.get("type") == "video":
            publisher = result.get("publisher")
            if isinstance(publisher, str) and publisher.strip():
                additional_fragments.append(f"Publisher: {publisher.strip()}")

        combined_parts: List[str] = []
        if title:
            combined_parts.append(title)
        if description:
            combined_parts.append(description)
        combined_parts.extend(additional_fragments)

        combined_text = " ".join(part for part in combined_parts if part).strip()

        if not combined_text:
            if config.verbose:
                logger.debug("Search result at index %d has no textual content. Skipping.", idx)
            continue

        normalized_items.append(
            {
                "title": title,
                "description": description,
                "text": combined_text,
                "url": result.get("url"),
                "type": result.get("type"),
                "source_rank": result.get("rank"),
                "match_score": result.get("match_score"),
                "raw": result,
            }
        )

    if not normalized_items:
        logger.debug("No searchable textual content extracted from search results.")
        return []

    retriever, reranker = load_models(config)

    chunk_database = create_corpus_chunks(normalized_items, config)
    if not chunk_database:
        logger.debug("No chunks created from search results.")
        return []

    corpus_embeddings = create_chunk_embeddings(
        chunk_database,
        retriever,
        verbose=config.verbose,
    )
    if corpus_embeddings is None:
        logger.debug("Failed to compute embeddings for search results.")
        return []

    ranked_candidates = rank_items(
        query,
        normalized_items,
        chunk_database,
        corpus_embeddings,
        retriever,
        reranker,
        config,
        final_top_k=final_top_k,
    )

    if save_path and ranked_candidates:
        path_obj = Path(save_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        serializable_payload = []
        for entry in ranked_candidates:
            item_payload = entry["item"].copy()
            serializable_payload.append(
                {
                    "rank": entry.get("rank"),
                    "aggregated_score": entry.get("aggregated_score"),
                    "num_chunks": entry.get("num_chunks"),
                    "chunk_scores": [float(score) for score in entry.get("chunk_scores", [])],
                    "item": item_payload,
                }
            )

        with path_obj.open("w", encoding="utf-8") as handle:
            json.dump(serializable_payload, handle, indent=2, ensure_ascii=False)

        if config.verbose:
            logger.debug("Saved ranked candidates to %s", path_obj)

    return ranked_candidates


def handle(
    query: str,
    search_payload: Union[Dict[str, Any], List[Dict[str, Any]]],
    output_folder: str,
    config: "PydanticCandidatesConfig | None" = None,
    *,
    final_top_k: int = 25,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """Main entry point for candidate ranking.

    Args:
        query: Search query string.
        search_payload: Search results from search module.
        output_folder: Directory for output files.
        config: CandidatesConfig instance or None for defaults.
        final_top_k: Number of top candidates to return.
        debug: Enable verbose debug output.

    Returns:
        List of ranked candidate dictionaries.
    """
    # Convert Pydantic config to dataclass if provided
    retrieval_config = RetrievalConfig(
        retriever_model=config.retriever_model if config else "nomic-ai/nomic-embed-text-v1.5",
        reranker_model=config.reranker_model if config else "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_chunk_tokens=config.max_chunk_tokens if config else 128,
        overlap_tokens=config.overlap_tokens if config else 32,
        retrieval_k=config.retrieval_k if config else 50,
        rerank_batch_size=config.rerank_batch_size if config else 32,
        aggregation_method=config.aggregation_method if config else "max",
        min_score_threshold=config.min_score_threshold if config else -5.0,
        score_margin_threshold=config.score_margin_threshold if config else None,
        verbose=debug,
    )

    save_path = Path(output_folder) / "candidates.json"

    return select_candidates_from_search_results(
        query,
        search_payload,
        config=retrieval_config,
        final_top_k=final_top_k,
        save_path=save_path,
    )


# --- Main execution ---
if __name__ == "__main__":
    
    # === Configuration ===
    # For maximum reliability in filtering relevant vs irrelevant:
    # - Use 'max' aggregation: if ANY chunk is highly relevant, the item matters
    # - Set a reasonable min_score_threshold to filter out low-quality matches
    # - Optionally use score_margin_threshold to only keep items close to the best match
    
    config = RetrievalConfig(
        retrieval_k=50,
        rerank_batch_size=32,
        aggregation_method='max',  # Best for finding truly relevant items
        min_score_threshold=-2.0,  # Filter out clearly irrelevant results
        score_margin_threshold=None,  # Set to e.g., 3.0 to only keep items within 3 points of best
        verbose=True,
        max_chunk_tokens=128,
        overlap_tokens=32
    )
    
    # --- Create a dummy JSON file with long text ---
    dummy_data = [
        {
            "title": "A History of Fictional Artificial Intelligence",
            "text": "The Master Control Program, or MCP, is a fictional software entity from the Tron franchise. It serves as the main antagonist, a power-hungry AI that rules the digital world. The MCP was originally a simple chess program, but it grew in power and intelligence, eventually betraying its user, Ed Dillinger. In the film, the protagonist Flynn must battle the MCP to restore freedom to the system. This story is a classic example of the 'rogue AI' trope in science fiction. It explores themes of control, freedom, and the unintended consequences of technology."
        },
        {
            "title": "Microsoft Certification Programs: An Overview",
            "text": "The Microsoft Certified Professional (MCP) program was a popular certification by Microsoft. It was designed to validate IT professional and developer technical expertise through rigorous, role-based exams. Earning an MCP certification showed that you were proficient in a specific Microsoft technology. Over the years, this program has evolved. It was eventually replaced by Microsoft's new role-based certifications, such as 'Azure Administrator Associate' or 'Microsoft 365 Developer Associate'. These new certifications are considered more relevant to modern cloud-based job roles."
        },
        {
            "title": "Anatomy of the Human Hand",
            "text": "The human hand is a complex structure. In medicine, MCP can stand for the Metacarpophalangeal joint. This is the knuckle joint at the base of each finger, connecting the metacarpal bones (in the palm) to the phalanges (finger bones). These joints are crucial for both gripping and making a fist. Conditions like arthritis often affect the MCP joints, causing pain and swelling. Understanding this anatomy is vital for surgeons and therapists."
        },
        {
            "title": "The Art of Roman Pasta",
            "text": "Italian cuisine is famous, but Roman pasta dishes are in a class of their own. 'Cacio e Pepe' is a simple but difficult dish, relying on only pasta, pecorino cheese, and black pepper. The key is creating a perfect, creamy emulsion using the starchy pasta water. Another classic is 'Carbonara', which uses egg yolk, guanciale (cured pork jowl), and pecorino. It's a rich and hearty meal. 'Amatriciana' is a red sauce, also using guanciale and pecorino, but with the addition of San Marzano tomatoes. These three dishes form a holy trinity of Roman pasta, beloved worldwide."
        },
        {
            "title": "Completely Unrelated Topic About Gardening",
            "text": "Gardening in urban environments requires careful planning and consideration of space constraints. Container gardening has become increasingly popular among city dwellers who want to grow their own vegetables and herbs. The key is selecting the right containers with proper drainage and using quality potting soil. Tomatoes, lettuce, and herbs like basil and parsley are excellent choices for beginners. Regular watering and adequate sunlight are essential for success."
        }
    ]
    
    json_filename = "search_data.json"
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(dummy_data, f, indent=2)
    print(f"Created dummy data file: {json_filename}\\n")
    
    # === Initialize System ===
    
    # 1. Load models
    retriever, reranker = load_models(config)
    
    # 2. Load data
    items_list = load_data(json_filename)
    
    if not items_list:
        print("No items loaded. Exiting.")
        exit(1)
    
    # 3. Create chunks
    chunk_database = create_corpus_chunks(items_list, config)
    
    if not chunk_database:
        print("No chunks created. Exiting.")
        exit(1)
    
    # 4. Pre-compute embeddings for all chunks
    corpus_embeddings = create_chunk_embeddings(
        chunk_database,
        retriever,
        verbose=config.verbose,
    )
    
    if corpus_embeddings is None:
        print("Failed to create embeddings. Exiting.")
        exit(1)
    
    # === Run Queries ===
    
    print("\\n" + "="*60)
    print("SYSTEM READY - RUNNING TEST QUERIES")
    print("="*60)
    
    # Query 1: Ambiguous query - should distinguish different meanings
    query1 = "what is MCP"
    ranked_results1 = rank_items(
        query1, items_list, chunk_database, corpus_embeddings,
        retriever, reranker, config, final_top_k=3
    )
    
    # Query 2: Specific to one item
    query2 = "who was the antagonist in Tron"
    ranked_results2 = rank_items(
        query2, items_list, chunk_database, corpus_embeddings,
        retriever, reranker, config, final_top_k=3
    )
    
    # Query 3: Specific to last item
    query3 = "how to make cacio e pepe"
    ranked_results3 = rank_items(
        query3, items_list, chunk_database, corpus_embeddings,
        retriever, reranker, config, final_top_k=3
    )
    
    # Query 4: Should filter out irrelevant content
    query4 = "explain quantum computing"
    ranked_results4 = rank_items(
        query4, items_list, chunk_database, corpus_embeddings,
        retriever, reranker, config, final_top_k=3
    )
    
    print("\\n" + "="*60)
    print("ALL QUERIES COMPLETED")
    print("="*60)