import json
import unicodedata
from unittest.mock import MagicMock, patch

import concurrent.futures
import google.api_core.exceptions
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agent.language import _sanitize, parse_instruction

OBJECT_TYPES = frozenset(["chair", "table", "lamp", "door", "box"])
_GOAL_KEYS = frozenset(("target", "attribute", "context", "error"))

# ── shared strategies ─────────────────────────────────────────────────────────

# Any valid target value: one of the known object types or None
valid_target = st.one_of(st.none(), st.sampled_from(sorted(OBJECT_TYPES)))

# Short strings within field length limits
short_str = st.one_of(st.none(), st.text(max_size=50))
context_str = st.one_of(st.none(), st.text(max_size=100))
error_str = st.one_of(st.none(), st.text(max_size=80))


@st.composite
def gemini_response_dict(draw):
    """
    Draw a dict that may have extra keys, missing optional keys, or all four
    keys — simulating the range of shapes Gemini might return.
    """
    target = draw(valid_target)
    include_attribute = draw(st.booleans())
    include_context = draw(st.booleans())
    include_error = draw(st.booleans())
    add_extra = draw(st.booleans())

    d: dict = {"target": target}
    if include_attribute:
        d["attribute"] = draw(short_str)
    if include_context:
        d["context"] = draw(context_str)
    if include_error:
        d["error"] = draw(error_str)
    if add_extra:
        extra_key = draw(st.text(min_size=1, max_size=20).filter(lambda k: k not in _GOAL_KEYS))
        d[extra_key] = draw(st.text(max_size=20))
    return d


@st.composite
def oversized_response_dict(draw):
    """
    Draw a dict with at least one field exceeding the allowed length limits,
    to confirm truncation logic fires.
    """
    target = draw(valid_target)
    # Force overlong strings on whichever field is drawn
    long_short = draw(st.text(min_size=51, max_size=120))
    long_context = draw(st.text(min_size=101, max_size=250))
    field = draw(st.sampled_from(["attribute", "context"]))
    d: dict = {"target": target}
    if field == "attribute":
        d["attribute"] = long_short
        d["context"] = draw(context_str)
    else:
        d["attribute"] = draw(short_str)
        d["context"] = long_context
    return d


@st.composite
def string_with_cc_injected(draw):
    """
    Draw a unicode string with at least one Cc codepoint injected at a
    random position, plus an arbitrary mix of normal characters.
    """
    # Draw a base text (no Cc)
    base = draw(st.text(alphabet=st.characters(blacklist_categories=("Cc",)), max_size=80))
    # Draw at least one Cc codepoint; Cc spans U+0000–U+001F and U+007F–U+009F
    cc_chars = draw(
        st.lists(
            st.one_of(
                st.integers(min_value=0x0000, max_value=0x001F),
                st.integers(min_value=0x007F, max_value=0x009F),
            ).map(chr),
            min_size=1,
            max_size=5,
        )
    )
    # Scatter the Cc characters at random positions in base
    result = list(base)
    for cc in cc_chars:
        pos = draw(st.integers(min_value=0, max_value=len(result)))
        result.insert(pos, cc)
    return "".join(result)


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_mock_model(text: str):
    """Return a mock GenerativeModel whose generate_content returns response.text = text."""
    mock_response = MagicMock()
    mock_response.text = text
    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_response
    return mock_model


def _call_parse(instruction: str, mock_model) -> dict:
    with patch("agent.language.genai.GenerativeModel", return_value=mock_model):
        return parse_instruction(instruction, OBJECT_TYPES)


# ── Property 8: Goal schema conformance ──────────────────────────────────────

# Feature: embodied-navigation-agent, Property 8: Goal schema conformance
@given(response_dict=gemini_response_dict())
@settings(max_examples=100)
def test_schema_conformance_exact_keys(response_dict):
    """parse_instruction always returns exactly the four goal keys, regardless of response shape."""
    mock_model = _make_mock_model(json.dumps(response_dict))
    result = _call_parse("go to the chair", mock_model)
    assert set(result.keys()) == _GOAL_KEYS


# Feature: embodied-navigation-agent, Property 8: Goal schema conformance
@given(response_dict=gemini_response_dict())
@settings(max_examples=100)
def test_schema_conformance_length_limits(response_dict):
    """target and attribute are capped at 50 chars; context at 100 chars."""
    mock_model = _make_mock_model(json.dumps(response_dict))
    result = _call_parse("go to the chair", mock_model)
    if isinstance(result["target"], str):
        assert len(result["target"]) <= 50
    if isinstance(result["attribute"], str):
        assert len(result["attribute"]) <= 50
    if isinstance(result["context"], str):
        assert len(result["context"]) <= 100


# Feature: embodied-navigation-agent, Property 8: Goal schema conformance
@given(response_dict=oversized_response_dict())
@settings(max_examples=100)
def test_schema_conformance_truncates_overlong_fields(response_dict):
    """Fields that arrive too long from Gemini are truncated, not errored."""
    mock_model = _make_mock_model(json.dumps(response_dict))
    result = _call_parse("go to the chair", mock_model)
    assert set(result.keys()) == _GOAL_KEYS
    if isinstance(result["attribute"], str):
        assert len(result["attribute"]) <= 50
    if isinstance(result["context"], str):
        assert len(result["context"]) <= 100


# ── Property 9: Error Goals for all API failure modes ────────────────────────

# Feature: embodied-navigation-agent, Property 9: Error Goals for all API failure modes
def test_timeout_deadline_exceeded():
    """DeadlineExceeded exception produces error='timeout' and target=None."""
    mock_model = MagicMock()
    mock_model.generate_content.side_effect = google.api_core.exceptions.DeadlineExceeded("timeout")
    result = _call_parse("go to the lamp", mock_model)
    assert result["target"] is None
    assert result["error"] == "timeout"


# Feature: embodied-navigation-agent, Property 9: Error Goals for all API failure modes
def test_timeout_futures_timeout():
    """concurrent.futures.TimeoutError also produces error='timeout' and target=None."""
    mock_model = MagicMock()
    mock_model.generate_content.side_effect = concurrent.futures.TimeoutError()
    result = _call_parse("go to the lamp", mock_model)
    assert result["target"] is None
    assert result["error"] == "timeout"


# Feature: embodied-navigation-agent, Property 9: Error Goals for all API failure modes
@given(
    status=st.one_of(
        st.integers(min_value=400, max_value=599),
        st.just(None),
    )
)
@settings(max_examples=100)
def test_api_error_produces_api_error_code(status):
    """GoogleAPICallError produces error starting with 'api_error:' and target=None."""
    exc = google.api_core.exceptions.GoogleAPICallError("api failure")
    if status is not None:
        exc.code = status
    else:
        # Make sure neither code nor http_status is set
        exc.code = None
        exc.http_status = None

    mock_model = MagicMock()
    mock_model.generate_content.side_effect = exc
    result = _call_parse("go to the table", mock_model)
    assert result["target"] is None
    assert result["error"].startswith("api_error:")


# Feature: embodied-navigation-agent, Property 9: Error Goals for all API failure modes
@given(bad_text=st.text(max_size=200).filter(lambda t: _is_not_valid_json(t)))
@settings(max_examples=100)
def test_non_json_response_gives_invalid_response(bad_text):
    """Non-JSON response text produces error='invalid_response' and target=None."""
    mock_model = _make_mock_model(bad_text)
    result = _call_parse("go to the door", mock_model)
    assert result["target"] is None
    assert result["error"] == "invalid_response"


# Feature: embodied-navigation-agent, Property 9: Error Goals for all API failure modes
@given(
    extra=st.fixed_dictionaries({
        "attribute": st.one_of(st.none(), st.text(max_size=30)),
        "context": st.one_of(st.none(), st.text(max_size=60)),
    })
)
@settings(max_examples=100)
def test_json_missing_target_gives_invalid_response(extra):
    """JSON that parses but has no 'target' key returns error='invalid_response'."""
    payload = {k: v for k, v in extra.items() if v is not None}
    mock_model = _make_mock_model(json.dumps(payload))
    result = _call_parse("go to the box", mock_model)
    assert result["target"] is None
    assert result["error"] == "invalid_response"


# Feature: embodied-navigation-agent, Property 9: Error Goals for all API failure modes
@given(unknown=st.text(min_size=1, max_size=40).filter(lambda t: t not in OBJECT_TYPES))
@settings(max_examples=100)
def test_unrecognized_target_gives_error(unknown):
    """Target value not in object_types returns error starting with 'unrecognized_target:'."""
    payload = {"target": unknown, "attribute": None, "context": None, "error": None}
    mock_model = _make_mock_model(json.dumps(payload))
    result = _call_parse("go to something", mock_model)
    assert result["target"] is None
    assert result["error"].startswith("unrecognized_target:")


# ── Property 10: Instruction sanitization removes Cc codepoints ──────────────

# Feature: embodied-navigation-agent, Property 10: Instruction sanitization removes Cc code points
@given(text=string_with_cc_injected())
@settings(max_examples=100)
def test_sanitize_removes_all_cc_codepoints(text):
    """_sanitize strips every Cc codepoint from the input string."""
    result = _sanitize(text)
    for ch in result:
        assert unicodedata.category(ch) != "Cc", (
            f"Cc character U+{ord(ch):04X} survived sanitization"
        )


# Feature: embodied-navigation-agent, Property 10: Instruction sanitization removes Cc code points
@given(text=string_with_cc_injected())
@settings(max_examples=100)
def test_sanitize_preserves_non_cc_chars_in_order(text):
    """_sanitize preserves all non-Cc characters and their relative order."""
    expected = [c for c in text if unicodedata.category(c) != "Cc"]
    result = list(_sanitize(text))
    assert result == expected


# Feature: embodied-navigation-agent, Property 10: Instruction sanitization removes Cc code points
@given(text=st.text(alphabet=st.characters(blacklist_categories=("Cc",)), max_size=200))
@settings(max_examples=100)
def test_sanitize_is_identity_on_clean_strings(text):
    """_sanitize is a no-op when the input has no Cc characters."""
    assert _sanitize(text) == text


# ── helper ────────────────────────────────────────────────────────────────────

def _is_not_valid_json(text: str) -> bool:
    """Return True when text is not valid JSON (used to filter strategy inputs)."""
    try:
        json.loads(text)
        return False
    except (json.JSONDecodeError, ValueError):
        return True
