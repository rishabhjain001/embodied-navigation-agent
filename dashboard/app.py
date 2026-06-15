import base64
import io
import os

import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Embodied Navigation Agent", layout="wide")

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# upscale tiny MiniGrid frames to something visible
_DISPLAY_SIZE = 420  # pixels


def _upscale(img: Image.Image) -> Image.Image:
    """Nearest-neighbour upscale to _DISPLAY_SIZE so grid cells stay crisp."""
    return img.resize((_DISPLAY_SIZE, _DISPLAY_SIZE), Image.NEAREST)


def _start_session(instruction: str) -> None:
    try:
        resp = requests.post(
            f"{BACKEND_URL}/session/start",
            json={"instruction": instruction},
            timeout=30,
        )
        if resp.ok:
            data = resp.json()
            st.session_state.session_id = data["session_id"]
            st.session_state.pov_frame_b64 = data["pov_frame"]
            st.session_state.goal = data["goal"]
            st.session_state.status = data["status"]
            st.session_state.step_log = []
            st.session_state.metrics = None
            st.session_state.error = None
        else:
            try:
                detail = resp.json().get("detail", "unknown error")
            except Exception:
                detail = resp.text or "unknown error"
            st.session_state.error = f"Backend error {resp.status_code}: {detail}"
    except requests.RequestException as exc:
        st.session_state.error = f"Connection error: {exc}"


def _do_step() -> None:
    sid = st.session_state.session_id
    try:
        resp = requests.post(f"{BACKEND_URL}/session/{sid}/step", timeout=30)
        if resp.ok:
            data = resp.json()
            st.session_state.pov_frame_b64 = data["pov_frame"]
            st.session_state.step_log.append({
                "step": data["step"],
                "action": data["action"],
                "reward": data["reward"],
                "agent_pos": data["agent_pos"],
            })
            st.session_state.metrics = data["metrics"]
            st.session_state.status = data["status"]
            st.session_state.error = None
        else:
            detail = resp.json().get("detail", "unknown error") if resp.content else "unknown error"
            st.session_state.error = f"Step error {resp.status_code}: {detail}"
    except requests.RequestException as exc:
        st.session_state.error = f"Connection error: {exc}"


def _do_run() -> None:
    sid = st.session_state.session_id
    try:
        resp = requests.post(f"{BACKEND_URL}/session/{sid}/run", timeout=90)
        if resp.ok:
            data = resp.json()
            st.session_state.step_log = data["step_log"]
            st.session_state.metrics = data["metrics"]
            st.session_state.status = data["status"]
            st.session_state.failure_reason = data.get("failure_reason")
            st.session_state.error = None
        else:
            detail = resp.json().get("detail", "unknown error") if resp.content else "unknown error"
            st.session_state.error = f"Run error {resp.status_code}: {detail}"
    except requests.RequestException as exc:
        st.session_state.error = f"Connection error: {exc}"


# Initialise all session state keys
_DEFAULTS = {
    "session_id": None,
    "pov_frame_b64": None,
    "goal": None,
    "status": None,
    "step_log": [],
    "metrics": None,
    "error": None,
    "loading": False,
    "failure_reason": None,
}
for _key, _val in _DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val

# ── Page title ─────────────────────────────────────────────────────────────────
st.title("🤖 Embodied Navigation Agent")
st.caption("Type a navigation instruction and watch the agent try to reach the target object.")

# ── Session startup ────────────────────────────────────────────────────────────
col_input, col_btn = st.columns([5, 1])
with col_input:
    instruction = st.text_input(
        "Navigation instruction",
        max_chars=500,
        placeholder="e.g. Go to the red chair",
        label_visibility="collapsed",
        disabled=st.session_state.loading,
    )
with col_btn:
    start_clicked = st.button("Start Session", disabled=st.session_state.loading, use_container_width=True)

if start_clicked:
    if instruction.strip():
        st.session_state.loading = True
        _start_session(instruction.strip())
        st.session_state.loading = False
    else:
        st.session_state.error = "Please enter a navigation instruction."

if st.session_state.error:
    st.error(st.session_state.error)

# ── Status banner (shown above everything when session ends) ───────────────────
if st.session_state.session_id is not None and st.session_state.status in ("success", "failure"):
    if st.session_state.status == "success":
        st.success("✅ **Success** — the agent reached the target!")
    elif st.session_state.failure_reason == "timeout":
        st.warning("⚠️ **Failed** — step budget exhausted (agent didn't find the target in time)")
    else:
        st.warning("⚠️ **Failed** — agent issued stop at the wrong location")

st.divider()

# ── Main content: two columns ──────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("Agent POV")
    if st.session_state.pov_frame_b64:
        img_bytes = base64.b64decode(st.session_state.pov_frame_b64)
        img = Image.open(io.BytesIO(img_bytes))
        big = _upscale(img)
        st.image(big, use_container_width=False)
        st.caption(
            f"Resolution: {img.width}×{img.height}px (MiniGrid partial view) "
            f"— displayed at {_DISPLAY_SIZE}×{_DISPLAY_SIZE}px"
        )
    else:
        st.info("Start a session to see the agent's view.")

    # Goal info
    if st.session_state.goal:
        g = st.session_state.goal
        if g.get("error"):
            st.error(f"Goal parse error: `{g['error']}`")
        else:
            parts = []
            if g.get("attribute"):
                parts.append(g["attribute"])
            if g.get("target"):
                parts.append(g["target"])
            goal_str = " ".join(parts) if parts else "unknown"
            st.markdown(f"**Goal:** navigate to the **{goal_str}**")
            if g.get("context"):
                st.caption(f"Context: {g['context']}")

with right:
    st.subheader("Controls")

    _session_active = (
        st.session_state.session_id is not None
        and st.session_state.status == "active"
        and not st.session_state.loading
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("⏭ Step", disabled=not _session_active, use_container_width=True):
            st.session_state.loading = True
            with st.spinner("Stepping…"):
                _do_step()
            st.session_state.loading = False
            st.rerun()
    with c2:
        if st.button("▶ Run to Completion", disabled=not _session_active, use_container_width=True):
            st.session_state.loading = True
            with st.spinner("Running to completion…"):
                _do_run()
            st.session_state.loading = False
            st.rerun()

    # Metrics
    if st.session_state.metrics:
        st.subheader("Metrics")
        m = st.session_state.metrics
        mc1, mc2 = st.columns(2)
        mc1.metric("Path Length", m.get("path_length", "—"))
        mc2.metric("Steps Taken", m.get("steps_to_goal") or "—")
        mc3, mc4 = st.columns(2)
        mc3.metric("Collisions", m.get("collision_count", "—"))
        eff = m.get("navigation_efficiency")
        mc4.metric("Efficiency", f"{eff:.2f}" if eff is not None else "—")
        sr = m.get("success_rate")
        st.metric("Session Success Rate", f"{sr:.0%}" if sr is not None else "—")

    # Step log
    if st.session_state.step_log:
        st.subheader("Step Log")
        df = pd.DataFrame(st.session_state.step_log)
        df.columns = [c.replace("_", " ").title() for c in df.columns]
        st.dataframe(df, use_container_width=False, height=250)
