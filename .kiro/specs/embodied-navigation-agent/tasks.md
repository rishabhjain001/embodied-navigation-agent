# Implementation Plan: Embodied Navigation Agent

## Overview

Implement the embodied navigation agent incrementally: project scaffolding first, then agent modules in dependency order (env → reward → vision → language → policy), then API layer (session → models → main), then training/eval scripts, then dashboard, and finally tests and documentation. Each layer builds on the previous so no code is orphaned.

---

## Tasks

- [x] 1. Project scaffolding
  - [x] 1.1 Write `requirements.txt` with all pinned versions grouped by section
    - Groups: `# RL`, `# Vision`, `# Simulation`, `# Language`, `# API`, `# Dashboard`
    - Pin: `stable-baselines3==2.3.2`, `gymnasium==0.29.1`, `torch==2.2.2`, `torchvision==0.17.2`, `git+https://github.com/openai/CLIP.git` (OpenAI CLIP — not on PyPI), `Pillow==10.3.0`, `minigrid==2.3.1`, `google-generativeai==0.5.4`, `fastapi==0.111.0`, `uvicorn==0.29.0`, `pydantic==2.7.1`, `python-dotenv==1.0.1`, `streamlit==1.35.0`, `requests==2.32.2`, `hypothesis`, `pytest`, `pytest-asyncio`
    - Note: CLIP must be listed as `git+https://github.com/openai/CLIP.git` in `requirements.txt` — there is no `clip` package on PyPI (that name belongs to an unrelated package)
    - _Requirements: 10.1_

  - [x] 1.2 Write `.env.example` with all required keys, blank values, one-line comment per key
    - Keys: `GEMINI_API_KEY`, `MODEL_PATH`, `GRID_SIZE`, `STEP_BUDGET`, `OBJECT_COUNT`, `REWARD_STEP_CLOSER`, `REWARD_STEP_AWAY`, `REWARD_COLLISION`, `REWARD_SUCCESS`, `REWARD_TIMEOUT`
    - _Requirements: 10.1_

  - [x] 1.3 Write `configs/default.yaml` with PPO hyperparameters, env settings, and reward values
    - Sections: `ppo` (learning_rate, n_steps, batch_size, n_epochs, gamma, gae_lambda, clip_range, total_timesteps), `env` (grid_size, object_count, step_budget), `reward` (step_closer, step_away, collision, success, timeout)
    - _Requirements: 10.1_

  - [x] 1.4 Write `.gitignore` covering Python artifacts, model weights, `.env`, virtual envs, and IDE files
    - Include: `__pycache__/`, `*.pyc`, `*.pth`, `*.zip`, `.env`, `models/*.zip`, `.venv/`, `.idea/`, `.vscode/`
    - _Requirements: 10.1_

- [x] 2. Environment module (`agent/env.py`)
  - [x] 2.1 Implement `make_env(grid_size, object_count, seed)` returning a wrapped MiniGrid env
    - Subclass `MiniGridEnv` for a single rectangular room; apply `RGBImgPartialObsWrapper`
    - Object types: `chair`, `table`, `lamp`, `door`, `box`; assign unique (type, color) combos each reset
    - Override `_gen_grid` to place exactly 5 objects at distinct cells and spawn agent at a random interior cell with a random cardinal orientation
    - Track `collision_count` in the `info` dict returned by `step()`; increment when `move_forward` leaves position unchanged
    - Action space: `Discrete(4)` → `{0: move_forward, 1: turn_left, 2: turn_right, 3: stop}`
    - Expose `env.object_types` as a frozenset of the 5 type strings
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7_

  - [x] 2.2 Implement `get_object_positions(env)` returning `{object_id: (row, col)}`
    - Read positions from env's ground-truth grid state; used only by `reward.py`
    - _Requirements: 1.2, 5.7_

  - [x]* 2.3 Write property tests for environment (P1–P4) in `tests/test_env.py`
    - **Property 1: Environment layout invariants on reset** — for any grid_size in [8,32] and any seed, `env.reset()` places exactly 5 objects at distinct cells with distinct (type, color) combos; agent at interior non-object cell with cardinal orientation
    - **Property 2: Consecutive sessions differ in layout** — for any env, two successive `env.reset()` calls without the same explicit seed produce different object position configurations
    - **Property 3: Collision keeps position and increments count** — when agent's forward cell is wall or occupied, `move_forward` leaves (row, col) and orientation unchanged and increments `info["collision_count"]` by 1
    - **Property 4: POV frame has valid shape and type on every step** — for any valid action, `obs["image"]` has shape `(H, W, 3)` with H, W > 0 and dtype `uint8`
    - Use `@settings(max_examples=100)` on each test
    - Tag each test: `# Feature: embodied-navigation-agent, Property {N}: {text}`
    - **Validates: Requirements 1.1, 1.2, 1.3, 1.5, 1.6, 1.7**

- [x] 3. Reward module (`agent/reward.py`)
  - [x] 3.1 Implement `euclidean_distance(a, b)` and `manhattan_distance(a, b)` helpers
    - _Requirements: 5.7_

  - [x] 3.2 Implement `compute_reward(action, prev_agent_pos, curr_agent_pos, target_pos, step_count, step_budget, done_stop)` returning `(reward, done)`
    - Accumulate distance component (+0.1 closer, -0.1 farther, 0.0 same), collision penalty (-0.5 when `curr == prev` and `action == move_forward`), terminal rewards (+10.0 success, -5.0 timeout)
    - `done=True` when stop issued OR `step_count >= step_budget`
    - Pure function: no env, session, or vision/language imports
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8_

  - [x]* 3.3 Write property tests for reward (P14–P15) in `tests/test_reward.py`
    - **Property 14: Stop action termination condition** — for any agent/target position pair, `stop` action → success iff `manhattan_distance <= 1`, failure otherwise; `done=True` always
    - **Property 15: Reward signal correctness** — for any position/action combination: move_forward decreasing distance → +0.1; increasing → -0.1; collision (pos unchanged) → additional -0.5; stop within 1 cell → +10.0 and done; `step_count >= step_budget` → -5.0 and done
    - **Validates: Requirements 5.1–5.6**

- [x] 4. Vision module (`agent/vision.py`)
  - [x] 4.1 Load `openai/clip-vit-base-patch32` at module level (singleton); implement `encode_frame(frame)` and `VisionModuleError`
    - `encode_frame`: resize any-size uint8 RGB array to 224×224 via CLIP preprocessor, return float32 `(512,)` L2-normalized embedding
    - Raise `VisionModuleError` on null input, wrong dtype, or internal CLIP error
    - Load with `clip.load("ViT-B/32", device="cpu")` — fail fast at import if weights unavailable
    - _Requirements: 2.1, 2.2, 2.5, 2.6_

  - [x] 4.2 Implement `compute_similarity(frame_embedding, text)` returning cosine similarity in `[-1.0, 1.0]`
    - Raise `ValueError` with descriptive message if `frame_embedding.shape != (512,)` or `text` is empty
    - _Requirements: 2.3, 2.4_

  - [x]* 4.3 Write property tests for vision module (P5–P7) in `tests/test_vision.py`
    - **Property 5: encode_frame produces fixed-size embedding** — for any valid RGB image (any size, uint8), `encode_frame()` returns float32 array of shape `(512,)`
    - **Property 6: compute_similarity output is bounded** — for any float32 `(512,)` array and any non-empty string, result is in `[-1.0, 1.0]`
    - **Property 7: Vision module raises on invalid inputs** — null/wrong-dtype input to `encode_frame` raises `VisionModuleError`; `(512,)` shape violation or empty text to `compute_similarity` raises `ValueError`
    - **Validates: Requirements 2.1, 2.3, 2.4, 2.6**

- [x] 5. Language module (`agent/language.py`)
  - [x] 5.1 Implement `_sanitize(text)` stripping Unicode Cc code points (U+0000–U+001F, U+007F–U+009F)
    - _Requirements: 3.7_

  - [x] 5.2 Implement `_build_prompt(instruction)` using the module-level `SYSTEM_PROMPT` constant
    - `SYSTEM_PROMPT` instructs Gemini to respond only with valid JSON matching the Goal schema
    - _Requirements: 3.8_

  - [x] 5.3 Implement `parse_instruction(instruction, object_types)` calling Gemini Pro API and returning a Goal dict
    - Sanitize instruction before sending; enforce 3-second timeout
    - Return dict with exactly keys `target`, `attribute`, `context`, `error`; discard extra fields from Gemini
    - Validate response is valid JSON with a `target` field before returning
    - Enforce field length limits: target/attribute ≤ 50 chars, context ≤ 100 chars
    - Return error Goals for: non-2xx HTTP → `"api_error:{status}"`, timeout → `"timeout"`, invalid JSON / missing target → `"invalid_response"`, target not in `object_types` → `"unrecognized_target:{token}"`
    - Never raises — all errors captured in `error` field
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9_

  - [x]* 5.4 Write property tests for language module (P8–P10) in `tests/test_language.py`
    - **Property 8: Goal schema conformance** — for any Gemini response shape (extra fields, missing optional fields), `parse_instruction()` returns dict with exactly keys `target`, `attribute`, `context`, `error`; values within length limits; mock Gemini API
    - **Property 9: Error Goals for all API failure modes** — for non-2xx responses, timeouts, non-JSON bodies, JSON missing `target`, and unrecognized targets, correct error code returned with `target=None`
    - **Property 10: Instruction sanitization removes Cc code points** — for any instruction with injected Cc codepoints, the string passed to Gemini contains none of those codepoints; all other characters preserved in order
    - **Validates: Requirements 3.2–3.9**

- [x] 6. Policy module (`agent/policy.py`)
  - [x] 6.1 Implement `build_obs_space()`, `encode_goal(goal)`, and `make_observation(frame_embedding, goal)`
    - `build_obs_space`: returns `Box(-inf, inf, (576,), float32)`
    - `encode_goal`: deterministic fixed-length float32 `(64,)` vector via seeded vocabulary hash; same function used at train and inference time
    - `make_observation`: concatenate `encode_goal(goal)` with `frame_embedding` → `(576,)`
    - _Requirements: 4.1, 4.6_

  - [x] 6.2 Implement `make_policy(env)` and `load_policy(model_path, env)`
    - `make_policy`: `PPO("MlpPolicy", env)` with hyperparams loaded from `configs/default.yaml`
    - `load_policy`: load PPO from `model_path`; raise `RuntimeError` with file path and failure reason on any error
    - _Requirements: 4.2, 4.4, 4.5_

  - [x] 6.3 Implement `predict_action(policy, obs)` returning action index in `{0, 1, 2, 3}`
    - _Requirements: 4.1_

  - [x]* 6.4 Write property tests for policy module (P11–P13) in `tests/test_policy.py`
    - **Property 11: Policy predict returns valid action for any observation** — for any float32 `(576,)` array, `predict_action()` returns integer in `{0, 1, 2, 3}`
    - **Property 12: Policy serialization round trip** — saving then loading a policy produces identical action predictions for any observation; use `tmp_path` fixture
    - **Property 13: Goal encoding is consistent and fixed-length** — for any Goal dict (including null fields), `encode_goal()` returns float32 `(64,)` array; calling twice with identical input returns element-wise equal arrays
    - **Validates: Requirements 4.1, 4.4, 4.6**

- [x] 7. Checkpoint — ensure all agent module tests pass
  - Run `pytest tests/test_env.py tests/test_reward.py tests/test_vision.py tests/test_language.py tests/test_policy.py -v`
  - Ensure all tests pass; ask the user if questions arise.

- [x] 8. API session state (`api/session.py`)
  - [x] 8.1 Define `StepRecord`, `EvaluationMetrics`, and `Session` dataclasses
    - `StepRecord`: `step`, `action`, `reward`, `agent_pos`
    - `EvaluationMetrics`: `path_length`, `steps_to_goal`, `collision_count`, `navigation_efficiency`, `success_rate` (nullable fields)
    - `Session`: all fields from design — `session_id`, `instruction`, `goal`, `env`, `policy`, `step_count`, `status`, `failure_reason`, `step_log`, `metrics`, `agent_start_pos`, `target_pos`, `collision_count`
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6_

  - [x] 8.2 Implement session store functions: `create_session`, `get_session`, `update_session`, `get_success_rate`
    - Module-level `_sessions` dict and `_experiment_stats` dict for aggregate success tracking
    - `get_success_rate`: returns `successes / total` or `None` if no sessions completed
    - _Requirements: 8.5, 6.5_

  - [x]* 8.3 Write property tests for evaluation metrics (P19) in `tests/test_metrics.py`
    - **Property 19: Evaluation metrics correctness** — for any session with a known action sequence and agent positions: `path_length` equals count of `move_forward` actions; `steps_to_goal` equals total action count through terminal step; `collision_count` equals count of `move_forward` where position unchanged; for successful sessions with `path_length > 0`, `navigation_efficiency = manhattan_distance(start, target) / path_length`; for zero path_length or zero manhattan distance, efficiency = 1.0; `success_rate` after N sessions = `successful / N`
    - **Validates: Requirements 8.1–8.5**

- [x] 9. API request/response schemas (`api/models.py`)
  - [x] 9.1 Implement all Pydantic v2 models: `StartSessionRequest`, `StartSessionResponse`, `StepResponse`, `StatusResponse`, `RunResponse`, `EvaluationMetricsModel`, `StepRecordModel`, `ConfigResponse`
    - `StartSessionRequest`: `instruction: str = Field(max_length=500)`
    - `StartSessionResponse`: `session_id`, `pov_frame` (base64 PNG string), `goal`, `status`
    - `StepResponse`: `step`, `action`, `reward`, `agent_pos`, `pov_frame`, `done`, `status`, `metrics`
    - `StatusResponse`: `session_id`, `status`, `failure_reason`, `step_log`, `metrics`
    - `EvaluationMetricsModel`: mirrors `EvaluationMetrics` dataclass with nullable fields
    - `ConfigResponse`: `step_budget`, `grid_size`, `object_count`, `model_path`
    - _Requirements: 6.6, 6.8_

  - [x]* 9.2 Write property tests for request validation (P18) in `tests/test_api.py`
    - **Property 18: Request validation returns 422 with field details** — for any malformed request body (missing required fields, wrong types, values exceeding max_length), Backend returns HTTP 422 with field-level error details; use FastAPI `TestClient`
    - **Validates: Requirements 6.8**

- [x] 10. FastAPI backend (`api/main.py`)
  - [x] 10.1 Implement startup handler that loads config from env vars / `.env` file, checks all required vars, loads CLIP via `vision.py` import, and loads PPO policy via `load_policy`
    - Collect all missing env vars in a single pass; log all of them and exit code 1 if any missing
    - If `load_policy` raises, log file path + reason and exit code 1
    - Expose loaded config values for `GET /config`
    - _Requirements: 4.5, 9.9, 10.1, 10.2, 10.3, 10.4, 10.5_

  - [x] 10.2 Implement `POST /session/start` endpoint
    - Accept `StartSessionRequest`; call `make_env`, `parse_instruction`, `create_session`; return `StartSessionResponse` with base64 POV frame
    - Return HTTP 500 with sanitized message if env init or goal parsing fails
    - _Requirements: 6.1, 6.6, 6.7_

  - [x] 10.3 Implement `POST /session/{session_id}/step` endpoint
    - Return HTTP 404 for unknown session ID; HTTP 409 if session already terminal
    - Execute one step: encode frame → `predict_action` → `env.step` → `compute_reward` → update session state and metrics
    - Return `StepResponse` with new POV frame as base64, updated metrics
    - _Requirements: 6.2, 6.3, 6.5, 6.6, 6.7_

  - [x] 10.4 Implement `GET /session/{session_id}/status` and `POST /session/{session_id}/run` endpoints
    - `status`: return HTTP 404 for unknown ID; return `StatusResponse` with step log and metrics
    - `run`: loop steps until `done`, enforce 60-second timeout via threading; return `StatusResponse` with final state
    - _Requirements: 6.3, 6.4, 6.5, 6.6_

  - [x] 10.5 Implement `GET /config` endpoint and global exception handler
    - `config`: return HTTP 503 if config not loaded; return `ConfigResponse` with non-secret values
    - Global handler: catch all unhandled exceptions → HTTP 500 `{"detail": "Internal server error"}`; no stack traces, no file paths
    - _Requirements: 6.7, 10.3, 10.4, 10.5_

  - [x]* 10.6 Write property tests for API session error responses (P16–P17) in `tests/test_api.py`
    - **Property 16: 404 for unknown session IDs** — for any UUID string not in session store, all session endpoints return HTTP 404
    - **Property 17: 409 for terminated sessions** — for any session that has reached a terminal state, `POST /session/{id}/step` returns HTTP 409
    - **Validates: Requirements 6.2, 6.5**

  - [x]* 10.7 Write property tests for config startup validation (P20) in `tests/test_config.py`
    - **Property 20: Missing config variables all reported at startup** — for any non-empty subset of required env vars that are absent, the startup validation collects and logs all missing variable names (not just the first), and signals non-zero exit
    - **Validates: Requirements 10.2**

- [x] 11. Checkpoint — ensure all API tests pass
  - Run `pytest tests/test_metrics.py tests/test_api.py tests/test_config.py -v`
  - Ensure all tests pass; ask the user if questions arise.

- [x] 12. Training script (`train.py`)
  - [x] 12.1 Implement config loading (YAML + env var override), CLI argument parsing, and `LoggingCallback`
    - CLI args: `--config` (default `configs/default.yaml`), `--model-path`, `--force-retrain`
    - Merge YAML values with env var overrides (env vars take precedence)
    - `LoggingCallback` extends `BaseCallback`: track per-episode success, compute mean reward, mean episode length, success rate over last 100 episodes
    - _Requirements: 9.1, 9.2, 9.7_

  - [x] 12.2 Implement training entry point: make env, load or create policy, train, save weights
    - If model file exists and not `--force-retrain`: call `load_policy` and continue training
    - If `--force-retrain` or no model file: call `make_policy` from scratch
    - Call `policy.learn(total_timesteps=..., callback=LoggingCallback)`; save to configured path
    - Log final mean reward, mean episode length, episode success rate on completion
    - Handle `RuntimeError` from `load_policy`, init failures, and I/O errors from `policy.save` — log descriptively and exit code 1
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9_

- [x] 13. Evaluation script (`scripts/evaluate.py`)
  - [x] 13.1 Implement standalone evaluation: load policy, run N episodes, print per-episode and aggregate metrics table
    - CLI args: `--model-path`, `--episodes` (default 10), `--config`, `--seed`
    - Load env and policy; run each episode to completion (step budget enforced)
    - Print per-episode table: episode number, success/failure, steps, path length, collisions, efficiency
    - Print aggregate row: success rate, mean steps, mean path length, mean collisions, mean efficiency
    - _Requirements: 9.1_

- [x] 14. Streamlit dashboard (`dashboard/app.py`)
  - [x] 14.1 Implement session startup UI: instruction text input (max 500 chars), "Start Session" button, and Backend HTTP call to `POST /session/start`
    - Store `session_id`, initial POV frame, goal, and status in `st.session_state`
    - Display error area if Backend returns 4xx/5xx
    - _Requirements: 7.1, 7.10_

  - [x] 14.2 Implement POV frame display, step log table, and evaluation metrics panel
    - POV frame: decode base64 PNG and render with `st.image`; update within 1 second of response
    - Step log: `st.dataframe` with columns step, action, reward, position; scrollable
    - Metrics panel: display path_length, steps_to_goal, collision_count, navigation_efficiency, success_rate
    - _Requirements: 7.2, 7.3, 7.4_

  - [x] 14.3 Implement "Step" and "Run to Completion" buttons with loading states and button disabling logic
    - "Step" calls `POST /session/{id}/step`; updates all panels
    - "Run to Completion" calls `POST /session/{id}/run`; updates all panels with final state
    - Disable both buttons when no active session (before first session or after session ends)
    - Show `st.spinner` while awaiting Backend response; disable buttons during loading
    - _Requirements: 7.5, 7.6, 7.7, 7.8_

  - [x] 14.4 Implement session end status banner
    - Display prominent banner above controls after session ends: "Success", "Failed — step budget exhausted", or "Failed — incorrect stop"
    - Keep dashboard interactive for new instruction input after session ends
    - _Requirements: 7.9_

- [x] 15. Integration tests (`tests/test_integration.py`)
  - [x]* 15.1 Write integration test: full episode run start → run → metrics correct
    - Start session with valid instruction, call `/run`, verify final `StatusResponse` has correct `status` and non-null `steps_to_goal`
    - _Requirements: 6.1, 6.3, 6.4, 8.2_

  - [x]* 15.2 Write integration test: policy save/load cycle with real SB3 API
    - Train a minimal policy (100 timesteps), save, load, verify prediction consistency
    - _Requirements: 4.4, 9.3_

  - [x]* 15.3 Write integration tests: Backend startup failure modes
    - Missing env vars → non-zero exit signal; corrupt model file → non-zero exit signal
    - _Requirements: 4.5, 10.2, 9.9_

  - [x]* 15.4 Write integration test: `POST /session/{id}/run` completes within 60 seconds for `step_budget <= 200`
    - _Requirements: 6.4_

- [x] 16. README (`README.md`)
  - [x] 16.1 Write README with project overview, architecture ASCII diagram, installation instructions, and citations
    - Opening paragraph: neutral tone describing what the system does
    - Architecture section: reproduce ASCII diagram from design doc
    - Setup section: exact commands for `pip install -r requirements.txt`, `.env` setup, training, running API and dashboard
    - Known limitations section: MVP scope boundaries (single room, 5 objects, no multi-step instructions, no multi-room)
    - Citations: MiniGrid (Chevalier-Boisvert et al.), CLIP (Radford et al., 2021), PPO (Schulman et al., 2017)
    - _Requirements: (documentation)_

- [x] 17. Final checkpoint — ensure all tests pass
  - Run `pytest tests/ -v --ignore=tests/test_integration.py` then `pytest tests/test_integration.py -v`
  - Ensure all tests pass; ask the user if questions arise.

---

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP delivery
- CLIP is not on PyPI — use `git+https://github.com/openai/CLIP.git` in `requirements.txt`; do not use `pip install clip` (that installs an unrelated package)
- Property tests use `@settings(max_examples=100)` and the tag format `# Feature: embodied-navigation-agent, Property {N}: {text}`
- Gemini API calls in language module tests should be mocked (no live API calls in unit/property tests)
- CLIP weights must be available in the environment for vision module tests; skip with `pytest.importorskip` if unavailable
- All reward computation is stateless — no env or session objects inside `reward.py`
- Config loading validates all missing env vars in a single pass before exiting (no partial startup)
- The `.env` file is loaded at startup via `python-dotenv`; env vars always take precedence over YAML
- Dashboard communicates with Backend exclusively via HTTP — no direct agent module imports
- Model weights are stored in `models/` (excluded from git via `.gitignore`)

---

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2", "1.3", "1.4"] },
    { "id": 1, "tasks": ["2.1", "2.2"] },
    { "id": 2, "tasks": ["2.3", "3.1"] },
    { "id": 3, "tasks": ["3.2", "4.1"] },
    { "id": 4, "tasks": ["3.3", "4.2", "5.1", "5.2"] },
    { "id": 5, "tasks": ["4.3", "5.3", "6.1"] },
    { "id": 6, "tasks": ["5.4", "6.2", "6.3"] },
    { "id": 7, "tasks": ["6.4", "8.1"] },
    { "id": 8, "tasks": ["8.2", "9.1"] },
    { "id": 9, "tasks": ["8.3", "9.2", "10.1"] },
    { "id": 10, "tasks": ["10.2", "10.3", "10.4"] },
    { "id": 11, "tasks": ["10.5", "12.1"] },
    { "id": 12, "tasks": ["10.6", "10.7", "12.2", "13.1"] },
    { "id": 13, "tasks": ["14.1", "15.2"] },
    { "id": 14, "tasks": ["14.2", "15.3"] },
    { "id": 15, "tasks": ["14.3", "15.1", "15.4"] },
    { "id": 16, "tasks": ["14.4"] },
    { "id": 17, "tasks": ["16.1"] }
  ]
}
```
