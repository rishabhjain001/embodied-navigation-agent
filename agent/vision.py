import hashlib
from functools import lru_cache

import numpy as np
import torch
from PIL import Image
import clip


class VisionModuleError(Exception):
    pass


# load once at import time — backend startup fails fast if weights are missing
_model, _preprocess = clip.load("ViT-B/32", device="cpu")

# Fixed random projection for fast inference without CLIP
# Same seed as training so observations are consistent
_PROJ_SEED = 42
_rng = np.random.default_rng(_PROJ_SEED)
_IMG_FLAT = 56 * 56 * 3
_PROJ_MATRIX = _rng.standard_normal((_IMG_FLAT, 512)).astype(np.float32) / np.sqrt(512)
USE_FAST_PROJECTION = False  # set to True to skip CLIP and use fast projection

# LRU cache for CLIP embeddings — avoids recomputing for repeated frames
_embedding_cache: dict[bytes, np.ndarray] = {}
_CACHE_MAX_SIZE = 512  # max number of cached frames


def _cache_key(frame: np.ndarray) -> bytes:
    # md5 of raw bytes is fast and collision-resistant enough for a cache key
    return hashlib.md5(frame.tobytes()).digest()


def encode_frame(frame: np.ndarray) -> np.ndarray:
    if frame is None:
        raise VisionModuleError("frame is None")
    if frame.dtype != np.uint8:
        raise VisionModuleError(f"expected uint8 frame, got {frame.dtype}")

    if USE_FAST_PROJECTION:
        # fast pixel projection — matches training observations
        flat = frame.astype(np.float32).reshape(-1) / 255.0
        if flat.shape[0] != _IMG_FLAT:
            img_resized = Image.fromarray(frame).resize((56, 56))
            flat = np.array(img_resized, dtype=np.float32).reshape(-1) / 255.0
        emb = flat @ _PROJ_MATRIX
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    # Check cache before running CLIP
    key = _cache_key(frame)
    if key in _embedding_cache:
        return _embedding_cache[key]

    try:
        img = Image.fromarray(frame)
        # _preprocess handles resize to 224×224 and normalization CLIP expects
        tensor = _preprocess(img).unsqueeze(0)
        with torch.no_grad():
            features = _model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)
        result = features.squeeze(0).cpu().numpy().astype(np.float32)
    except VisionModuleError:
        raise
    except Exception as exc:
        raise VisionModuleError(f"CLIP encode failed: {exc}") from exc

    # Store in cache, evict oldest if full
    if len(_embedding_cache) >= _CACHE_MAX_SIZE:
        _embedding_cache.pop(next(iter(_embedding_cache)))
    _embedding_cache[key] = result
    return result


def compute_similarity(frame_embedding: np.ndarray, text: str) -> float:
    """Returns cosine similarity in [-1, 1]. Raises ValueError on shape mismatch or empty text."""
    if frame_embedding.shape != (512,):
        raise ValueError(
            f"frame_embedding must have shape (512,), got {frame_embedding.shape}"
        )
    if text == "":
        raise ValueError("text must not be empty")

    tokens = clip.tokenize([text])
    with torch.no_grad():
        text_features = _model.encode_text(tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    text_emb = text_features.squeeze(0).cpu().numpy().astype(np.float32)

    similarity = float(np.dot(frame_embedding, text_emb))
    return float(np.clip(similarity, -1.0, 1.0))
