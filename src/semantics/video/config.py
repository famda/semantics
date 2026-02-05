"""Video processing configuration classes.

Defines Pydantic models for each video processing module, centralizing
defaults and enabling YAML-based configuration overrides.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import yaml
from pydantic import BaseModel, Field


class DownloadConfig(BaseModel):
    """Configuration for YouTube/URL video downloading."""

    max_height: int = Field(
        default=720,
        description="Maximum video height (e.g., 360, 720, 1080) when downloading from a URL.",
    )
    filename_template: str = Field(
        default="%(title)s_%(id)s.%(ext)s",
        description="yt-dlp filename template for downloaded videos.",
    )


class FramesConfig(BaseModel):
    """Configuration for frame extraction from video."""

    target_fps: Optional[float] = Field(
        default=None,
        description="Target frames per second for extraction. None uses source FPS.",
    )
    parallel_jobs: Optional[int] = Field(
        default=None,
        description="Number of parallel FFmpeg jobs for frame extraction. None auto-detects.",
    )


class SegmentsConfig(BaseModel):
    """Configuration for adaptive keyframe/segment extraction."""

    target_detection_fps: Optional[float] = Field(
        default=12.0,
        description="FPS used for scene detection (downsampled from source). Lower = faster.",
    )
    include_last_frame: bool = Field(
        default=True,
        description="Always include the last frame of the video as a keyframe.",
    )


class ClusteringConfig(BaseModel):
    """Configuration for MobileNet-based frame clustering and keyframe selection."""

    mn_threshold: float = Field(
        default=0.995,
        description="MobileNet cosine similarity threshold for cluster merging.",
    )
    mn_eps: float = Field(
        default=0.2,
        description="DBSCAN epsilon for MobileNet feature clustering.",
    )
    mn_min_samples: int = Field(
        default=3,
        description="DBSCAN min_samples for MobileNet clustering.",
    )
    fps: float = Field(
        default=1.0,
        description="Frame sampling rate for clustering analysis.",
    )
    keyframes: bool = Field(
        default=True,
        description="Enable keyframe selection from clusters.",
    )
    kf_eps: float = Field(
        default=0.18,
        description="DBSCAN epsilon for keyframe sub-clustering.",
    )
    kf_min_samples: int = Field(
        default=1,
        description="DBSCAN min_samples for keyframe sub-clustering.",
    )
    kf_hamming_frac: float = Field(
        default=0.30,
        description="Fraction of Hamming distance for perceptual hash filtering.",
    )
    kf_require_both: bool = Field(
        default=True,
        description="Require both DBSCAN and Hamming criteria for keyframe selection.",
    )
    save_keyframes: bool = Field(
        default=True,
        description="Save selected keyframes to disk as images.",
    )
    save_clusters: bool = Field(
        default=False,
        description="Save all cluster frames to disk (can be large).",
    )


class CaptionsConfig(BaseModel):
    """Configuration for Florence-2 based image captioning and analysis."""

    model_id: str = Field(
        default="microsoft/Florence-2-large-ft",
        description="HuggingFace model ID for Florence-2 captioning.",
    )
    precision: str = Field(
        default="fp16",
        description="Model precision: 'fp16', 'fp32', or 'bf16'.",
    )
    default_queries: str = Field(
        default="person. car. dog. cat. bicycle. chair. book. phone. text.",
        description="Default object queries for visual grounding (dot-separated).",
    )
    run_ocr: bool = Field(
        default=False,
        description="Enable OCR text extraction from frames.",
    )
    run_objects: bool = Field(
        default=False,
        description="Enable object detection in frames.",
    )
    run_visual_grounding: bool = Field(
        default=False,
        description="Enable visual grounding for object queries.",
    )


class ObjectsConfig(BaseModel):
    """Configuration for YOLO-based object detection, pose estimation, and face recognition."""

    detection_model: str = Field(
        default="yolo26x.pt",
        description="YOLO model for object detection (e.g., yolo11s.pt, yolo26s.pt).",
    )
    segmentation_model: str = Field(
        default="yolo26x-seg.pt",
        description="YOLO model for instance segmentation (e.g., yolo11s-seg.pt, yolo26s-seg.pt).",
    )
    pose_model: str = Field(
        default="yolo26x-pose.pt",
        description="YOLO model for pose estimation (e.g., yolo11s-pose.pt, yolo26s-pose.pt).",
    )
    object_conf_threshold: float = Field(
        default=0.80,
        description="Confidence threshold for YOLO object detection.",
    )
    iou_match_threshold: float = Field(
        default=0.5,
        description="IoU threshold for matching detections across frames.",
    )
    keypoint_conf_threshold: float = Field(
        default=0.6,
        description="Confidence threshold for pose keypoint detection.",
    )
    face_conf_threshold: float = Field(
        default=0.9,
        description="Confidence threshold for face detection.",
    )
    embedding_model_name: str = Field(
        default="Facenet512",
        description="DeepFace embedding model for face recognition.",
    )
    face_detect_min_side: int = Field(
        default=720,
        description="Minimum image side length for face detection upscaling.",
    )
    face_detect_max_scale: float = Field(
        default=2.0,
        description="Maximum scale factor for face detection upscaling.",
    )
    detector_backend: str = Field(
        default="retinaface",
        description="DeepFace detector backend: 'retinaface', 'mtcnn', 'opencv', etc.",
    )
    clip_model_name: str = Field(
        default="ViT-B/32",
        description="OpenAI CLIP model for object embedding and similarity.",
    )
    cluster_base_eps: float = Field(
        default=0.35,
        description="Base DBSCAN epsilon for object/face clustering.",
    )
    cluster_min_samples: int = Field(
        default=1,
        description="DBSCAN min_samples for object clustering.",
    )
    cluster_min_attempts: int = Field(
        default=4,
        description="Minimum clustering attempts for adaptive epsilon.",
    )
    keyframe_eps: float = Field(
        default=0.12,
        description="DBSCAN epsilon for keyframe extraction from object clusters.",
    )
    keyframe_min_samples: int = Field(
        default=1,
        description="DBSCAN min_samples for keyframe extraction.",
    )
    keyframe_hamming_frac: float = Field(
        default=0.30,
        description="Hamming distance fraction for perceptual hash keyframe filtering.",
    )
    keyframe_require_both: bool = Field(
        default=True,
        description="Require both DBSCAN and Hamming criteria for keyframe selection.",
    )


class TilesConfig(BaseModel):
    """Configuration for creating video tile/mosaic images."""

    columns: int = Field(
        default=3,
        description="Number of columns in the tile grid.",
    )
    rows: int = Field(
        default=3,
        description="Number of rows in the tile grid.",
    )
    final_tile_width: Optional[int] = Field(
        default=None,
        description="Final tile width in pixels. None preserves original dimensions.",
    )
    final_tile_height: Optional[int] = Field(
        default=None,
        description="Final tile height in pixels. None preserves original dimensions.",
    )
    background_color: Tuple[int, int, int] = Field(
        default=(0, 255, 0),
        description="RGB background color for empty tile spaces.",
    )


class ScenesConfig(BaseModel):
    """Configuration for PySceneDetect scene splitting."""

    detector_type: str = Field(
        default="content",
        description="Scene detector type: 'content' or 'threshold'.",
    )
    threshold: float = Field(
        default=27.0,
        description="Detection threshold for scene changes.",
    )
    min_scene_len: int = Field(
        default=15,
        description="Minimum scene length in frames.",
    )
    use_codec_copy: bool = Field(
        default=False,
        description="Use codec copy for faster scene splitting (may cause issues at boundaries).",
    )


class ActionsConfig(BaseModel):
    """Configuration for action recognition in video.
    
    Uses transformer-based models (TimeSformer, VideoMAE) to recognize
    human actions in video clips with temporal understanding.
    
    The module uses motion-based activity scanning to intelligently select
    clips for analysis, focusing on areas with actual motion/activity.
    """

    model: str = Field(
        default="MCG-NJU/videomae-base-finetuned-kinetics",
        description="HuggingFace model ID for action recognition. Options: MCG-NJU/videomae-large-finetuned-kinetics (recommended, 16 frames), MCG-NJU/videomae-base-finetuned-kinetics (faster), facebook/timesformer-base-finetuned-k400 (fastest, 8 frames).",
    )
    num_frames: int = Field(
        default=16,
        description="Number of frames to sample per clip. Must match model requirements (8 for TimeSformer, 16 for VideoMAE).",
    )
    frame_sample_rate: int = Field(
        default=4,
        description="Sample every n-th frame. Temporal span = num_frames * frame_sample_rate frames. Higher values capture longer actions (default 4 = ~2.1s clips at 30fps).",
    )
    conf_threshold: float = Field(
        default=0.40,
        description="Minimum confidence threshold for action predictions (0.0 - 1.0). Lower values capture more actions but may include false positives.",
    )
    top_k: int = Field(
        default=3,
        description="Number of top action predictions to return per clip.",
    )
    batch_size: int = Field(
        default=8,
        description="Batch size for processing video clips. Higher = faster with more GPU parallelism (default 8 for RTX 6GB+).",
    )
    save_clips: bool = Field(
        default=True,
        description="Save video clips for each detected action to actions/clips/ folder.",
    )
    padding: float = Field(
        default=1.0,
        description="Seconds of padding for temporal context. Applied to activity segments during scanning and to saved action clips.",
    )
    clip_overlap: float = Field(
        default=0.75,
        description="Overlap ratio (0.0-1.0) between consecutive clips in activity segments. 0.75 = 75% overlap for thorough coverage.",
    )
    scan_fps: float = Field(
        default=1.0,
        description="Frames per second to analyze during activity pre-scan. Higher = more accurate but slower.",
    )
    motion_threshold: float = Field(
        default=0.02,
        description="Minimum motion ratio (0.0-1.0) to consider a frame as active. 0.02 = 2% of pixels changed.",
    )
    min_activity_duration: float = Field(
        default=0.5,
        description="Minimum duration in seconds for an activity segment to be analyzed.",
    )
    merge_gap: float = Field(
        default=1.0,
        description="Maximum gap in seconds between activity segments to merge them.",
    )


class OcrConfig(BaseModel):
    """Configuration for EasyOCR-based text extraction."""

    confidence_threshold: float = Field(
        default=70.0,
        description="Minimum confidence (0-100) for text detection.",
    )
    lang: str = Field(
        default="en",
        description="Language code for OCR (e.g., 'en', 'de', 'fr').",
    )
    save_images: bool = Field(
        default=True,
        description="Save annotated images with detected text bounding boxes.",
    )


class ClassificationConfig(BaseModel):
    """Configuration for YOLO-based image classification."""

    model: str = Field(
        default="yolo26s-cls.pt",
        description="YOLO classification model (e.g., yolo11s-cls.pt, yolo11m-cls.pt).",
    )
    conf_threshold: float = Field(
        default=0.25,
        description="Minimum confidence threshold for predictions (0.0 - 1.0).",
    )
    top_k: int = Field(
        default=5,
        description="Number of top predictions to return per frame.",
    )


class NerConfig(BaseModel):
    """Configuration for Named Entity Recognition on video captions."""

    model_name: str = Field(
        default="Jean-Baptiste/roberta-large-ner-english",
        description="HuggingFace model ID for NER",
    )
    device: Optional[str] = Field(
        default=None, description="Device override (cuda/cpu/None=auto)"
    )
    batch_size: int = Field(default=8, description="Batch size for inference")
    confidence_threshold: float = Field(
        default=0.92, description="Minimum confidence for entity detection"
    )
    aggregate_strategy: str = Field(
        default="simple",
        description="Token aggregation strategy: simple, first, average, max",
    )
    caption_field: str = Field(
        default="caption_more_detailed",
        description="Caption field to use for NER: caption, caption_detailed, caption_more_detailed",
    )


class VideoConfig(BaseModel):
    """Root configuration for all video processing modules."""

    download: DownloadConfig = Field(
        default_factory=DownloadConfig,
        description="YouTube/URL download settings.",
    )
    frames: FramesConfig = Field(
        default_factory=FramesConfig,
        description="Frame extraction settings.",
    )
    segments: SegmentsConfig = Field(
        default_factory=SegmentsConfig,
        description="Adaptive keyframe/segment extraction settings.",
    )
    clustering: ClusteringConfig = Field(
        default_factory=ClusteringConfig,
        description="Frame clustering and keyframe selection settings.",
    )
    captions: CaptionsConfig = Field(
        default_factory=CaptionsConfig,
        description="Florence-2 captioning settings.",
    )
    objects: ObjectsConfig = Field(
        default_factory=ObjectsConfig,
        description="Object detection and face recognition settings.",
    )
    tiles: TilesConfig = Field(
        default_factory=TilesConfig,
        description="Tile/mosaic generation settings.",
    )
    scenes: ScenesConfig = Field(
        default_factory=ScenesConfig,
        description="Scene splitting settings.",
    )
    actions: ActionsConfig = Field(
        default_factory=ActionsConfig,
        description="Action recognition settings.",
    )
    ocr: OcrConfig = Field(
        default_factory=OcrConfig,
        description="OCR text extraction settings.",
    )
    classification: ClassificationConfig = Field(
        default_factory=ClassificationConfig,
        description="Image classification settings.",
    )
    ner: NerConfig = Field(
        default_factory=NerConfig,
        description="Named Entity Recognition settings.",
    )


def _normalize_video_payload(payload: dict) -> dict:
    """Normalize config payload by extracting 'video' key if present."""
    data = dict(payload or {})
    if "video" in data and isinstance(data["video"], dict):
        data = dict(data["video"])
    return data


def _parse_model(model_cls: type[BaseModel], data: dict) -> BaseModel:
    """Parse data into a Pydantic model (compatible with v1 and v2)."""
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(data)
    return model_cls.parse_obj(data)


def load_video_config(path: str) -> VideoConfig:
    """Load video configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        VideoConfig instance with merged defaults and file values.
    """
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    payload = _normalize_video_payload(raw)
    return _parse_model(VideoConfig, payload)
