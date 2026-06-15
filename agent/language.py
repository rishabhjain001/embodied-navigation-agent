import concurrent.futures
import json
import os
import unicodedata

import google.generativeai as genai
import google.api_core.exceptions

SYSTEM_PROMPT = (
    "You are a navigation instruction parser. "
    "Respond ONLY with a JSON object — no prose, no markdown, no explanation. "
    "The JSON must have exactly these four keys:\n"
    '  "target"    — the object type to navigate to (string, max 50 chars) or null\n'
    '  "attribute" — a color or modifier for the target (string, max 50 chars) or null\n'
    '  "context"   — a spatial phrase providing extra context (string, max 100 chars) or null\n'
    '  "error"     — an error message if the instruction is unparseable (string) or null\n'
    "Do not include any keys other than these four. "
    "Do not wrap the JSON in a code block or add any surrounding text."
)

_GOAL_KEYS = ("target", "attribute", "context", "error")

# Known colors for the keyword fallback parser
_COLORS = ("red", "green", "blue", "purple", "yellow", "grey", "gray")


def keyword_parse(instruction: str, object_types: frozenset[str]) -> dict:
    """Simple keyword matcher used as a fallback when the Gemini API is unavailable.

    Scans the instruction for a known object type and optional color so the
    system stays usable without a live API call (e.g. during rate limits).
    """
    text = instruction.lower()
    target = next((obj for obj in object_types if obj in text), None)
    if target is None:
        return _error_goal(f"unrecognized_target:{instruction[:30]}")
    attribute = next((c for c in _COLORS if c in text), None)
    if attribute == "gray":
        attribute = "grey"
    return {"target": target, "attribute": attribute, "context": None, "error": None}


def _sanitize(text: str) -> str:
    return "".join(c for c in text if unicodedata.category(c) != "Cc")


def _build_prompt(instruction: str) -> str:
    return f"{SYSTEM_PROMPT}\n\nInstruction: {_sanitize(instruction)}"


def _error_goal(error_code: str) -> dict:
    return {"target": None, "attribute": None, "context": None, "error": error_code}


def parse_instruction(instruction: str, object_types: frozenset[str]) -> dict:
    """Returns a Goal dict. Never raises — all errors land in the 'error' field."""
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-2.0-flash")

    try:
        response = model.generate_content(
            _build_prompt(instruction),
            request_options={"timeout": 3},
        )
        text = response.text
    except (google.api_core.exceptions.DeadlineExceeded, concurrent.futures.TimeoutError):
        return _error_goal("timeout")
    except google.api_core.exceptions.GoogleAPICallError as exc:
        status_code = getattr(exc, "code", None)
        if status_code is None:
            status_code = getattr(exc, "http_status", 0)
        return _error_goal(f"api_error:{status_code}")

    # Validate JSON and presence of required 'target' field
    try:
        parsed = json.loads(text)
        if "target" not in parsed:
            return _error_goal("invalid_response")
    except (json.JSONDecodeError, TypeError):
        return _error_goal("invalid_response")

    # Pick only the four expected keys; missing optional ones default to None
    goal = {k: parsed.get(k) for k in _GOAL_KEYS}

    # Enforce field length limits (truncate, don't error)
    for field in ("target", "attribute"):
        if isinstance(goal[field], str) and len(goal[field]) > 50:
            goal[field] = goal[field][:50]
    if isinstance(goal["context"], str) and len(goal["context"]) > 100:
        goal["context"] = goal["context"][:100]

    # Validate target against known object types
    token = goal["target"]
    if token is not None and token not in object_types:
        return _error_goal(f"unrecognized_target:{token}")

    return goal
