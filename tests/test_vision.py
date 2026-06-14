import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from agent.vision import VisionModuleError, compute_similarity, encode_frame


# ── shared strategies ────────────────────────────────────────────────────────

@st.composite
def rgb_image(draw):
    """Draw a random uint8 RGB image of varying height and width."""
    h = draw(st.integers(min_value=1, max_value=224))
    w = draw(st.integers(min_value=1, max_value=224))
    pixels = draw(st.binary(min_size=h * w * 3, max_size=h * w * 3))
    return np.frombuffer(pixels, dtype=np.uint8).reshape(h, w, 3)


@st.composite
def normalized_float32_embedding(draw):
    """Draw a random float32 (512,) array and L2-normalize it to mimic real embeddings."""
    raw = draw(
        st.lists(
            st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=512,
            max_size=512,
        )
    )
    arr = np.array(raw, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm == 0.0:
        arr[0] = 1.0
        norm = 1.0
    return arr / norm


# ── Property 5 ───────────────────────────────────────────────────────────────

# Feature: embodied-navigation-agent, Property 5: encode_frame produces fixed-size embedding
@given(frame=rgb_image())
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.data_too_large])
def test_encode_frame_returns_fixed_shape(frame):
    """encode_frame always returns float32 (512,) regardless of input image dimensions."""
    embedding = encode_frame(frame)
    assert embedding.shape == (512,)
    assert embedding.dtype == np.float32


# ── Property 6 ───────────────────────────────────────────────────────────────

# Feature: embodied-navigation-agent, Property 6: compute_similarity output is bounded
@given(
    embedding=normalized_float32_embedding(),
    text=st.text(min_size=1, max_size=200),
)
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.large_base_example],
)
def test_compute_similarity_is_bounded(embedding, text):
    """compute_similarity result is always within [-1.0, 1.0]."""
    result = compute_similarity(embedding, text)
    assert -1.0 <= result <= 1.0


# ── Property 7 — invalid inputs to encode_frame ──────────────────────────────

# Feature: embodied-navigation-agent, Property 7: Vision module raises on invalid inputs
def test_encode_frame_raises_on_none():
    """encode_frame(None) raises VisionModuleError."""
    with pytest.raises(VisionModuleError):
        encode_frame(None)


# Feature: embodied-navigation-agent, Property 7: Vision module raises on invalid inputs
@given(
    h=st.integers(min_value=1, max_value=64),
    w=st.integers(min_value=1, max_value=64),
)
@settings(max_examples=100)
def test_encode_frame_raises_on_wrong_dtype(h, w):
    """encode_frame raises VisionModuleError when the frame dtype is not uint8."""
    bad_frame = np.zeros((h, w, 3), dtype=np.float32)
    with pytest.raises(VisionModuleError):
        encode_frame(bad_frame)


# ── Property 7 — invalid inputs to compute_similarity ────────────────────────

# Feature: embodied-navigation-agent, Property 7: Vision module raises on invalid inputs
@given(
    size=st.integers(min_value=0, max_value=1024).filter(lambda n: n != 512),
)
@settings(max_examples=100)
def test_compute_similarity_raises_on_wrong_shape(size):
    """compute_similarity raises ValueError when embedding shape is not (512,)."""
    bad_emb = np.zeros((size,), dtype=np.float32)
    with pytest.raises(ValueError):
        compute_similarity(bad_emb, "a chair")


# Feature: embodied-navigation-agent, Property 7: Vision module raises on invalid inputs
def test_compute_similarity_raises_on_empty_text():
    """compute_similarity raises ValueError when text is empty."""
    valid_emb = np.zeros((512,), dtype=np.float32)
    valid_emb[0] = 1.0
    with pytest.raises(ValueError):
        compute_similarity(valid_emb, "")
