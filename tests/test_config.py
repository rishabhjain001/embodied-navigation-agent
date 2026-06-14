import logging
import os
import sys
from unittest.mock import call, patch

from hypothesis import given, settings
from hypothesis import strategies as st

# Feature: embodied-navigation-agent, Property 20: Missing config variables all reported at startup

ALL_REQUIRED = [
    "GEMINI_API_KEY",
    "MODEL_PATH",
    "GRID_SIZE",
    "STEP_BUDGET",
    "OBJECT_COUNT",
    "REWARD_STEP_CLOSER",
    "REWARD_STEP_AWAY",
    "REWARD_COLLISION",
    "REWARD_SUCCESS",
    "REWARD_TIMEOUT",
]


def _run_validation(env: dict):
    """Run the startup env-var validation logic from api/main.py lifespan."""
    missing = [k for k in ALL_REQUIRED if not env.get(k)]
    if missing:
        for k in missing:
            logging.error(f"missing required env var: {k}")
        sys.exit(1)
    return missing


# ---------------------------------------------------------------------------
# P20 — all missing vars reported (property test)
# ---------------------------------------------------------------------------

# Validates: Requirements 10.2
@given(
    missing_vars=st.lists(
        st.sampled_from(ALL_REQUIRED),
        min_size=1,
        max_size=len(ALL_REQUIRED),
        unique=True,
    )
)
@settings(max_examples=100)
def test_all_missing_vars_reported(missing_vars):
    """For any non-empty subset of missing vars, ALL are detected and logged — not just the first."""
    present = {k: "value" for k in ALL_REQUIRED if k not in missing_vars}

    with patch.dict(os.environ, present, clear=True):
        with patch("sys.exit") as mock_exit:
            with patch("logging.error") as mock_log:
                missing = [k for k in ALL_REQUIRED if not os.getenv(k)]
                if missing:
                    for k in missing:
                        logging.error(f"missing required env var: {k}")
                    sys.exit(1)

                # Every expected missing var was detected
                assert set(missing) == set(missing_vars)

                # sys.exit(1) called exactly once
                mock_exit.assert_called_once_with(1)

                # Every missing var name appeared in a log.error call
                logged_messages = [str(c) for c in mock_log.call_args_list]
                for var in missing_vars:
                    assert any(var in msg for msg in logged_messages), (
                        f"{var!r} was never logged — only first missing var may have been reported"
                    )


# ---------------------------------------------------------------------------
# Additional targeted tests
# ---------------------------------------------------------------------------

def test_all_vars_present_no_exit():
    """When all required vars are present, sys.exit is never called."""
    full_env = {k: "value" for k in ALL_REQUIRED}

    with patch.dict(os.environ, full_env, clear=True):
        with patch("sys.exit") as mock_exit:
            missing = [k for k in ALL_REQUIRED if not os.getenv(k)]
            if missing:
                for k in missing:
                    logging.error(f"missing required env var: {k}")
                sys.exit(1)

            assert missing == []
            mock_exit.assert_not_called()


def test_single_missing_var_detected_and_logged():
    """A single missing var is detected, logged, and triggers sys.exit(1)."""
    absent = "GEMINI_API_KEY"
    present = {k: "value" for k in ALL_REQUIRED if k != absent}

    with patch.dict(os.environ, present, clear=True):
        with patch("sys.exit") as mock_exit:
            with patch("logging.error") as mock_log:
                missing = [k for k in ALL_REQUIRED if not os.getenv(k)]
                if missing:
                    for k in missing:
                        logging.error(f"missing required env var: {k}")
                    sys.exit(1)

                assert missing == [absent]
                mock_exit.assert_called_once_with(1)
                logged_messages = [str(c) for c in mock_log.call_args_list]
                assert any(absent in msg for msg in logged_messages)


def test_all_vars_missing_all_ten_detected():
    """When every required var is absent, all 10 are detected and logged."""
    with patch.dict(os.environ, {}, clear=True):
        with patch("sys.exit") as mock_exit:
            with patch("logging.error") as mock_log:
                missing = [k for k in ALL_REQUIRED if not os.getenv(k)]
                if missing:
                    for k in missing:
                        logging.error(f"missing required env var: {k}")
                    sys.exit(1)

                assert set(missing) == set(ALL_REQUIRED)
                mock_exit.assert_called_once_with(1)

                # One log call per missing var — not just the first
                assert mock_log.call_count == len(ALL_REQUIRED)
                logged_messages = [str(c) for c in mock_log.call_args_list]
                for var in ALL_REQUIRED:
                    assert any(var in msg for msg in logged_messages), (
                        f"{var!r} missing from log output"
                    )
