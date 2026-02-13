import json
import os
from typing import List, Optional, Sequence

import numpy as np

AUDIO_FILE_TYPES = ["mp3", "wav", "flac", "ogg", "m4a"]
VIDEO_FILE_TYPES = ["mp4", "mkv", "avi", "mov", "webm"]
MODEL_FILE_TYPES = ["glb", "gltf", "fbx", "obj", "stl"]
DOCUMENT_FILE_TYPES = ["pdf", "docx", "pptx", "txt", "md"]

LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
    "yue": "cantonese",
}

whisper_langs = sorted(LANGUAGES.keys()) + sorted([name.title() for name in LANGUAGES.values()])

# Define mapping from specific YAMNet classes to our generic categories
# You can get the full list of 521 classes from the model documentation or its class map csv:
# https://storage.googleapis.com/audioset/yamnet/yamnet_class_map.csv
# This mapping is a simplified example and can be refined!
AUDIO_CLASSIFICATION_CATEGORY_MAP = {
    "Speech/Dialogue": [
        "Speech", "Child speech, kid speaking", "Conversation", "Narration, monologue",
        "Babbling", "Speech synthesizer", "Shout", "Bellow", "Whoop", "Yell", "Whispering"
    ],
    "Music": [
        "Music", "Musical instrument", "Plucked string instrument", "Guitar", "Banjo",
        "Sitar", "Mandolin", "Zither", "Ukulele", "Keyboard (musical)", "Piano", "Electric piano",
        "Organ", "Harpsichord", "Synthesizer", "Percussion", "Drum kit", "Drum machine", "Snare drum",
        "Bass drum", "Cymbal", "Hi-hat", "Wood block", "Tambourine", "Rattle (instrument)",
        "Maraca", "Gong", "Tubular bells", "Mallet percussion", "Marimba, xylophone", "Glockenspiel",
        "Vibraphone", "Steelpan", "Orchestra", "Brass instrument", "Trumpet", "Trombone", "French horn",
        "Tuba", "Bowed string instrument", "Violin, fiddle", "Viola", "Cello", "Double bass",
        "Wind instrument, woodwind instrument", "Flute", "Clarinet", "Saxophone", "Oboe", "Bassoon",
        "Harmonica", "Accordion", "Bagpipes", "Singing", "Choir", "Yodeling", "Chant", "Mantra",
        "Child singing", "Synthetic singing", "Rapping", "Humming", "Groan", "Moan", "Sigh",
        "Beatboxing", "Scat singing", "Speech singing", "Music genre", "Pop music", "Hip hop music",
        "Rock music", "Heavy metal", "Punk rock", "Grunge", "Progressive rock", "Rock and roll",
        "Psychedelic rock", "Blues", "Rhythm and blues", "Soul music", "Reggae", "Country", "Swing music",
        "Bluegrass", "Funk", "Folk music", "Middle Eastern music", "Jazz", "Disco", "Classical music",
        "Opera", "Electronic music", "House music", "Techno", "Dubstep", "Drum and bass", "Electronica",
        "Ambient music", "Trance music", "Music of Latin America", "Salsa music", "Flamenco", "Blues",
        "Music of Asia", "Music of Africa", "Indian classical music", "Bollywood", "Ska",
        "Gospel music", "Theme music", "Jingle (music)", "Music for children", "New-age music",
        "Independent music", "Song" # Adding Song as explicitly music
    ],
    # Everything else falls into Mixed/Other
}

VIDEO_OBJECT_DETECTION_CATEGORY_MAP = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "airplane": 4,
    "bus": 5,
    "train": 6,
    "truck": 7,
    "boat": 8,
    "traffic light": 9,
    "fire hydrant": 10,
    "stop sign": 11,
    "parking meter": 12,
    "bench": 13,
    "bird": 14,
    "cat": 15,
    "dog": 16,
    "horse": 17,
    "sheep": 18,
    "cow": 19,
    "elephant": 20,
    "bear": 21,
    "zebra": 22,
    "giraffe": 23,
    "backpack": 24,
    "umbrella": 25,
    "handbag": 26,
    "tie": 27,
    "suitcase": 28,
    "frisbee": 29,
    "skis": 30,
    "snowboard": 31,
    "sports ball": 32,
    "kite": 33,
    "baseball bat": 34,
    "baseball glove": 35,
    "skateboard": 36,
    "surfboard": 37,
    "tennis racket": 38,
    "bottle": 39,
    "wine glass": 40,
    "cup": 41,
    "fork": 42,
    "knife": 43,
    "spoon": 44,
    "bowl": 45,
    "banana": 46,
    "apple": 47,
    "sandwich": 48,
    "orange": 49,
    "broccoli": 50,
    "carrot": 51,
    "hot dog": 52,
    "pizza": 53,
    "donut": 54,
    "cake": 55,
    "chair": 56,
    "couch": 57,
    "potted plant": 58,
    "bed": 59,
    "dining table": 60,
    "toilet": 61,
    "tv": 62,
    "laptop": 63,
    "mouse": 64,
    "remote": 65,
    "keyboard": 66,
    "cell phone": 67,
    "microwave": 68,
    "oven": 69,
    "toaster": 70,
    "sink": 71,
    "refrigerator": 72,
    "book": 73,
    "clock": 74,
    "vase": 75,
    "scissors": 76,
    "teddy bear": 77,
    "hair drier": 78,
    "toothbrush": 79
}

VIDEO_OBJECT_DETECTION_KEYPOINT_MAP = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16
}

VIDEO_OBJECT_DETECTION_KEYPOINT_GROUPING = {
    "person": [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ],
    "car": [
        "front_left_wheel", "front_right_wheel", "rear_left_wheel", "rear_right_wheel",
        "front_bumper", "rear_bumper", "left_mirror", "right_mirror"
    ],
    # Add more classes and their keypoints as needed    
}

_FRAME_TIME_KEYS = (
    "pts_time",
    "pkt_pts_time",
    "pkt_dts_time",
    "best_effort_timestamp_time",
)


def select_frame_indices(metadata: Sequence[dict], target_fps: Optional[float]) -> List[int]:
    """Return frame indices sampled at the requested FPS based on metadata timestamps."""
    if not metadata:
        return []

    try:
        fps_value = float(target_fps) if target_fps is not None else None
    except (TypeError, ValueError):
        fps_value = None

    if fps_value is None or fps_value <= 0:
        return list(range(len(metadata)))

    frame_times: List[float] = []
    fallback_step = 1.0 / fps_value if fps_value > 0 else 1.0

    for entry in metadata:
        time_value = None
        for key in _FRAME_TIME_KEYS:
            if key in entry and entry[key] is not None:
                time_value = entry[key]
                break

        numeric_time: Optional[float]
        if time_value is not None:
            try:
                numeric_time = float(time_value)
            except (TypeError, ValueError):
                numeric_time = None
        else:
            numeric_time = None

        if numeric_time is None:
            numeric_time = (frame_times[-1] + fallback_step) if frame_times else 0.0
        elif frame_times and numeric_time < frame_times[-1]:
            numeric_time = frame_times[-1]

        frame_times.append(numeric_time)

    if not frame_times:
        return []

    total_duration = frame_times[-1]
    selected: List[int] = []
    idx = 0
    current_time = 0.0
    step = 1.0 / fps_value
    epsilon = 1e-6

    while idx < len(frame_times) and current_time <= total_duration + epsilon:
        while idx < len(frame_times) and frame_times[idx] < current_time - epsilon:
            idx += 1
        if idx >= len(frame_times):
            break
        if not selected or selected[-1] != idx:
            selected.append(idx)
        current_time += step

    if not selected and metadata:
        selected.append(0)

    return selected


def select_frames_by_fps(json_file_path, frames_folder=".", fps=None):

    with open(json_file_path, "r") as f:
        data = json.load(f)

    frames = data.get("frames", [])
    if not frames:
        return []

    indices = select_frame_indices(frames, fps)
    return [os.path.join(frames_folder, f"{idx:08d}.png") for idx in indices]


def l2_normalize_rows(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        return arr
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def estimate_dbscan_eps(vectors: np.ndarray, base_eps: float, min_samples: int) -> float:
    try:
        from sklearn.neighbors import NearestNeighbors
    except Exception:
        return base_eps

    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < max(min_samples + 1, 4):
        return base_eps

    k = min(max(min_samples * 2, 4), arr.shape[0])
    try:
        nn = NearestNeighbors(metric="cosine", n_neighbors=k)
        nn.fit(arr)
        distances, _ = nn.kneighbors(arr)
    except Exception:
        return base_eps

    if distances.size == 0:
        return base_eps

    neighbor_dists = distances[:, 1:]
    if neighbor_dists.size == 0:
        return base_eps

    kth = neighbor_dists[:, -1]
    finite = kth[np.isfinite(kth)]
    if finite.size == 0:
        return base_eps

    p_low = float(np.percentile(finite, 40))
    p_high = float(np.percentile(finite, 75))
    candidate = max(p_low, min(p_high, float(base_eps)))
    candidate = max(0.02, min(candidate, float(base_eps)))
    return candidate


def coerce_int(value) -> Optional[int]:
    """Convert *value* to ``int``, handling floats and ``None``.

    Returns ``None`` when the conversion is not possible.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None
