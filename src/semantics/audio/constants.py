from typing import Literal

AUDIO_FILE_TYPES = ['mp3', 'wav', 'flac', 'aac', 'ogg', 'm4a']

TRANSCRIPTION_ALLOWED_MODELS = Literal[
    "tiny.en", "tiny", "base.en", "base", "small.en", "small", "medium.en", "medium",
    "large-v1", "large-v2", "large-v3", "large", "distil-large-v2", "distil-medium.en",
    "distil-small.en", "distil-large-v3", "distil-large-v3.5", "large-v3-turbo", "turbo"
]