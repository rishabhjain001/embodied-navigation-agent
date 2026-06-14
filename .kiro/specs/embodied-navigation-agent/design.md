# Design Document: Embodied Navigation Agent

## Overview

The embodied navigation agent is a reinforcement learning system that follows natural language instructions in a MiniGrid grid world. A user submits a free-text instruction such as "Go to the red chair." The system parses that instruction into a structured goal using Gemini Pro, encodes the agent's first-person visual observation using CLIP, and feeds the combined representation into a PPO policy that selects the next navigation action.

The MVP scope is a single rectangular room containing exactly 5 objects. The agent has four actions: `move_forward`, `turn_left`, `turn_right`, and `stop`. A session ends when the agent issues `stop` (success or failure) or exhausts the step budget (timeout). A FastAPI backend orchestrates all components; a Streamlit dashboard provides a live view of the agent's POV, step log, and evaluation metrics.

The system is designed to be straightforward and flat: no abstract base classes, no plugin registries, no over-engineered pipelines. Each module does one thing and exposes simple functions.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User / Dashboard                            │
│                     (Streamlit  dashboard/app.py)                   │
└─────────────────────┬───────────────────────────────────────────────┘
                      │  HTTP (JSON + base64 PNG)
┌─────────────────────▼───────────────────────────────────────────────┐
│                       FastAPI Backend                               │
│                        api/main.py                                  │
│                                                                     │
│   POST /session/start          GET /session/{id}/status             │
│   POST /session/{id}/step      POST /session/{id}/run               │
│   GET  /config                                                      │
│                                                                     │
│   api/session.py  ──  session store (dict, in-memory)               │
│   api/models.py   ──  Pydantic request / response schemas           │
└──────┬────────────────────────┬────────────────────────┬────────────┘
       │                        │                        │
┌──────▼──────┐        ┌────────▼───────┐       ┌───────▼──────────┐
│  Language   │        │    Vision      │       │   Navigation     │
│  Module     │        │    Module      │       │   Policy         │
│ language.py │        │  vision.py     │       │   policy.py      │
│             │        │                │       │                  │
│ parse_instr │        │ encode_frame   │       │ load_policy      │
│ → Goal dict │        │ compute_sim    │       │ predict_action   │
└──────┬──────┘        └────────┬───────┘       └───────┬──────────┘
       │ Gemini Pro API         │ CLIP                  │ SB3 PPO
       │ (external)             │ (local weights)       │ (MlpPolicy)
       │                        │                       │
       └──────────────────┬─────┘                       │
                          │   Observation (concat)       │
                          └─────────────────────────────┘
                                        │
                             ┌──────────▼──────────┐
                             │   Environment        │
                             │   agent/env.py       │
                             │                      │
                             │ MiniGrid + RGB        │
                             │ ImgPartialObs         │
                             │ Wrapper               │
                             └──────────┬────────────┘
                                        │ step obs, done, info
                             ┌──────────▼────────────┐
                             │   Reward System        │
                             │   agent/reward.py      │
                             │                        │
                             │ compute_reward(...)    │
                             └────────────────────────┘
```

**Data flow for a single step:**

1. Dashboard calls `POST /session/{id}/step`
2. Backend retrieves session state from `api/session.py`
3. `vision.py` encodes the current POV frame → 512-dim embedding
4. `policy.py` concatenates embedding + goal encoding → calls PPO predict → action
5. `env.py` steps the MiniGrid environment → new obs, done flag
6. `reward.py` computes scalar reward from env positions
7. Backend updates session state, Step_Log, and Evaluation_Metrics
8. Response JSON (action, reward, new POV frame as base64, metrics) returned to Dashboard

---

## Components and Interfaces

### agent/env.py

The environment wraps MiniGrid's `MiniGridEnv` with `RGBImgPartialObsWrapper`. It exposes a standard Gymnasium interface with one additional method for reading ground-truth object positions (used by the reward system).

```python
# Key functions
def make_env(grid_size: int, object_count: int, seed: int | None = None) -> gym.Env
    # Returns a wrapped MiniGrid env. RGBImgPartialObsWrapper is applied internally.
    # object_types is accessible as env.object_types (frozenset of strings).

def get_object_positions(env: gym.Env) -> dict[str, tuple[int, int]]
    # Returns {object_id: (row, col)} using env ground-truth grid state.
    # Used only by reward.py — never by vision.py or language.py.
```

Object layout is randomized each `env.reset()` call via MiniGrid's built-in seed mechanism. The five object types are: `chair`, `table`, `lamp`, `door`, `box`. Each object is assigned one of MiniGrid's built-in colors (red, green, blue, purple, yellow, grey) such that the (type, color) combination is unique per session.

The `collision_count` is tracked in the `info` dict returned by `env.step()`.

**Action space:** `Discrete(4)` mapping to `{0: move_forward, 1: turn_left, 2: turn_right, 3: stop}`.

**Observation space:** Dict with key `"image"` → shape `(obs_height, obs_width, 3)`, dtype `uint8`. The observation height/width is determined by `RGBImgPartialObsWrapper` defaults (typically 56×56 for a 7-cell partial view field).

---

### agent/vision.py

Loads `openai/clip-vit-base-patch32` once at import time (module-level singleton). Exposes two functions.

```python
def encode_frame(frame: np.ndarray) -> np.ndarray
    # frame: uint8 RGB array, any size — resized to 224×224 internally via CLIP preprocess.
    # Returns: float32 array of shape (512,), L2-normalized.
    # Raises: VisionModuleError on null input, wrong dtype, or internal CLIP error.

def compute_similarity(frame_embedding: np.ndarray, text: str) -> float
    # frame_embedding: float32 array of shape (512,).
    # text: non-empty string.
    # Returns: cosine similarity in [-1.0, 1.0].
    # Raises: ValueError if frame_embedding.shape != (512,) or text is empty string.

class VisionModuleError(Exception):
    pass
```

The CLIP model and preprocessor are loaded at module import via `clip.load("ViT-B/32", device="cpu")`. No lazy loading — Backend startup will fail fast if weights are unavailable.

---

### agent/language.py

Calls Gemini Pro API to parse a free-text instruction into a structured Goal dict. All network I/O is synchronous (no async) to keep integration with SB3 training loop simple.

```python
def parse_instruction(instruction: str, object_types: frozenset[str]) -> dict
    # instruction: raw user text, max 500 chars.
    # object_types: set of valid target types for the current env.
    # Returns: Goal dict with keys: target (str|None), attribute (str|None),
    #          context (str|None), error (str|None).
    # Never raises — all errors are returned in the 'error' field.

def _sanitize(text: str) -> str
    # Strips Unicode Cc category code points (U+0000–U+001F, U+007F–U+009F).
    # Internal helper, not exported.

def _build_prompt(instruction: str) -> str
    # Constructs the Gemini system + user prompt from a fixed template.
    # Internal helper.
```

The fixed system prompt instructs Gemini to respond only with a JSON object matching the Goal schema. The prompt is a module-level constant `SYSTEM_PROMPT`.

Timeout is enforced via Python's `requests` timeout parameter (or `google-generativeai` equivalent). If the API call raises a timeout exception, `parse_instruction` catches it and returns `{"target": null, "error": "timeout"}`.

Error handling hierarchy (all returned, never raised):
- HTTP non-2xx → `{"target": null, "error": "api_error:{status_code}"}`
- Timeout → `{"target": null, "error": "timeout"}`
- Response not valid JSON or missing `target` → `{"target": null, "error": "invalid_response"}`
- Target not in `object_types` → `{"target": null, "error": "unrecognized_target:{token}"}`

---

### agent/policy.py

Wraps Stable-Baselines3 PPO. Handles observation space construction, goal encoding, and action prediction.

```python
GOAL_ENCODING_DIM = 64   # fixed-length goal vector dimension
OBS_DIM = 512 + GOAL_ENCODING_DIM  # 576 total

def build_obs_space() -> gym.spaces.Box
    # Returns Box(low=-inf, high=inf, shape=(OBS_DIM,), dtype=float32).
    # Called by make_policy() and train.py to ensure consistency.

def encode_goal(goal: dict) -> np.ndarray
    # goal: {target, attribute, context, error}.
    # Returns float32 array of shape (GOAL_ENCODING_DIM,).
    # Uses a fixed vocabulary hash to produce a deterministic, fixed-length vector.
    # Same function used during training and inference.

def make_observation(frame_embedding: np.ndarray, goal: dict) -> np.ndarray
    # Concatenates encode_goal(goal) with frame_embedding → shape (OBS_DIM,).

def make_policy(env: gym.Env) -> PPO
    # Constructs a fresh PPO(MlpPolicy, env) with hyperparams from configs/default.yaml.

def load_policy(model_path: str, env: gym.Env) -> PPO
    # Loads PPO from model_path. Raises RuntimeError with descriptive message on failure.
    # Caller (Backend startup) is responsible for catching and exiting.

def predict_action(policy: PPO, obs: np.ndarray) -> int
    # Returns action index in {0, 1, 2, 3}.
```

The goal encoding uses a simple but deterministic scheme: target, attribute, and context strings are each hashed into a fixed sub-vector via a seeded word embedding lookup table (small, initialized at module load). This produces a consistent fixed-length representation without requiring a separate language model.

---

### agent/reward.py

Stateless reward computation. Takes explicit position arguments — no env or session objects.

```python
def compute_reward(
    action: int,
    prev_agent_pos: tuple[int, int],
    curr_agent_pos: tuple[int, int],
    target_pos: tuple[int, int],
    step_count: int,
    step_budget: int,
    done_stop: bool,
) -> tuple[float, bool]
    # Returns (reward, done).
    # done is True when stop action issued OR step_count >= step_budget.
    # reward accumulates: distance_component + collision_penalty + terminal_reward.

def euclidean_distance(a: tuple[int, int], b: tuple[int, int]) -> float
    # Standard 2D Euclidean distance helper.

def manhattan_distance(a: tuple[int, int], b: tuple[int, int]) -> int
    # Used by metrics computation.
```

Reward logic (all additive):
- If `action == move_forward`:
  - `prev_dist > curr_dist` → +0.1
  - `prev_dist < curr_dist` → -0.1
  - `prev_dist == curr_dist` → 0.0
  - If `curr_agent_pos == prev_agent_pos` (collision) → additional -0.5
- If `action == stop`:
  - `manhattan_distance(curr_agent_pos, target_pos) <= 1` → +10.0, done=True (success)
  - Otherwise → 0.0, done=True (failure)
- If `step_count >= step_budget`: → -5.0, done=True (timeout)

---

### api/session.py

In-memory session store. Holds all state for active and completed sessions.

```python
@dataclass
class Session:
    session_id: str             # UUID4 string
    instruction: str
    goal: dict                  # parsed Goal
    env: gym.Env
    policy: PPO
    step_count: int
    status: str                 # "active" | "success" | "failure"
    failure_reason: str | None  # "timeout" | "incorrect_stop" | None
    step_log: list[StepRecord]
    metrics: EvaluationMetrics
    agent_start_pos: tuple[int, int]
    target_pos: tuple[int, int]
    collision_count: int

@dataclass
class StepRecord:
    step: int
    action: str
    reward: float
    agent_pos: tuple[int, int]

@dataclass
class EvaluationMetrics:
    path_length: int            # count of move_forward actions
    steps_to_goal: int | None   # null until terminal
    collision_count: int
    navigation_efficiency: float | None  # null until terminal success
    success_rate: float | None  # null until at least 1 session complete

# Module-level store
_sessions: dict[str, Session] = {}
_experiment_stats = {"total": 0, "successes": 0}

def create_session(instruction, goal, env, policy) -> Session
def get_session(session_id: str) -> Session | None
def update_session(session: Session) -> None
def get_success_rate() -> float | None
```

Sessions are never deleted during a Backend run (in-memory only, no persistence). The `_experiment_stats` dict tracks aggregate success rate across all sessions in the current Backend run.

---

### api/models.py

Pydantic v2 models for all request/response schemas.

```python
class StartSessionRequest(BaseModel):
    instruction: str = Field(max_length=500)

class StartSessionResponse(BaseModel):
    session_id: str
    pov_frame: str          # base64-encoded PNG
    goal: dict
    status: str

class StepResponse(BaseModel):
    step: int
    action: str
    reward: float
    agent_pos: list[int]    # [row, col]
    pov_frame: str          # base64-encoded PNG
    done: bool
    status: str
    metrics: EvaluationMetricsModel

class StatusResponse(BaseModel):
    session_id: str
    status: str
    failure_reason: str | None
    step_log: list[StepRecordModel]
    metrics: EvaluationMetricsModel

class EvaluationMetricsModel(BaseModel):
    path_length: int
    steps_to_goal: int | None
    collision_count: int
    navigation_efficiency: float | None
    success_rate: float | None

class ConfigResponse(BaseModel):
    step_budget: int
    grid_size: int
    object_count: int
    model_path: str
```

---

### api/main.py

FastAPI app with five endpoints. Startup handler loads CLIP weights (via `vision.py` import) and the PPO policy.

```python
@app.post("/session/start") -> StartSessionResponse
@app.post("/session/{session_id}/step") -> StepResponse
@app.get("/session/{session_id}/status") -> StatusResponse
@app.post("/session/{session_id}/run") -> StatusResponse
@app.get("/config") -> ConfigResponse
```

Global exception handler catches unhandled exceptions and returns HTTP 500 with a generic message — no tracebacks, no file paths. 404 and 409 are raised explicitly via `HTTPException`.

The `/session/{id}/run` endpoint loops steps internally until `done == True`, with a 60-second overall timeout enforced via `asyncio.wait_for` (or equivalent threading timeout if running sync).

---

### dashboard/app.py

Streamlit app. Communicates with the Backend exclusively via HTTP. No direct imports of agent modules.

Key UI elements:
- Instruction text input (max 500 chars) + "Start Session" button
- POV frame image display (updated each step)
- Step log table (scrollable, columns: step, action, reward, position)
- Evaluation metrics panel (5 values)
- "Step" button and "Run to Completion" button
- Status banner (success / failure / timeout)
- Error area for Backend error responses
- Loading spinner (st.spinner) while awaiting Backend

Session state is stored in `st.session_state` (session_id, current status, step_log, metrics).

---

### configs/default.yaml

```yaml
ppo:
  learning_rate: 0.0003
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  total_timesteps: 500000

env:
  grid_size: 10          # overridden by GRID_SIZE env var
  object_count: 5        # overridden by OBJECT_COUNT env var
  step_budget: 200       # overridden by STEP_BUDGET env var

reward:
  step_closer: 0.1
  step_away: -0.1
  collision: -0.5
  success: 10.0
  timeout: -5.0
```

Environment variables take precedence over YAML values at runtime. `train.py` and the Backend both load this file at startup.

---

### train.py

Standalone training script. Accepts CLI args `--config` (path to YAML), `--model-path` (output file), `--force-retrain` flag.

```python
# Entry point flow:
# 1. Load config (YAML + env vars)
# 2. Make env via make_env()
# 3. If model file exists and not --force-retrain: load_policy()
# 4. Else: make_policy()
# 5. policy.learn(total_timesteps=config.total_timesteps, callback=LoggingCallback)
# 6. policy.save(model_path)
# 7. Log final mean reward, mean episode length, episode success rate over last 100 eps
```

`LoggingCallback` extends SB3's `BaseCallback` to track per-episode success and compute the final stats.

---

### scripts/evaluate.py

Standalone evaluation script. Loads a saved policy and runs N episodes, printing per-episode and aggregate metrics.

---

## Data Models

### Goal

```python
{
    "target": str | None,     # object type, max 50 chars; null on error
    "attribute": str | None,  # color or modifier, max 50 chars
    "context": str | None,    # spatial phrase, max 100 chars; optional
    "error": str | None       # error code; null on success
}
```

Valid error codes: `"api_error:{status}"`, `"timeout"`, `"invalid_response"`, `"unrecognized_target:{token}"`.

### Observation

A flat `float32` numpy array of shape `(576,)`:
- `[0:512]` — CLIP frame embedding from `encode_frame()`
- `[512:576]` — goal encoding from `encode_goal()`

The 576-dim flat vector is the only input to the PPO `MlpPolicy`. The observation space is defined in `policy.py` as `Box(-inf, inf, (576,), float32)`.

### StepRecord

```python
{
    "step": int,
    "action": str,       # "move_forward" | "turn_left" | "turn_right" | "stop"
    "reward": float,
    "agent_pos": [int, int]   # [row, col]
}
```

### EvaluationMetrics

```python
{
    "path_length": int,                  # total move_forward actions
    "steps_to_goal": int | None,         # total actions; null until terminal
    "collision_count": int,
    "navigation_efficiency": float | None,  # manhattan_start_target / path_length; null until success
    "success_rate": float | None         # successes / total; null until >= 1 session done
}
```

### Session (internal)

See `api/session.py` above. Session IDs are UUID4 strings. Sessions are keyed by session ID in the module-level `_sessions` dict. The `env` object is kept alive for the duration of the session.

---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Environment layout invariants on reset

*For any* valid grid size in [8, 32] and any random seed, when a session begins (`env.reset()` is called), the environment SHALL contain exactly 5 objects, all at distinct grid cells, each with a distinct (type, color) combination, and the agent SHALL be placed at an interior cell not occupied by any object with one of the four cardinal orientations.

**Validates: Requirements 1.1, 1.2, 1.6**

---

### Property 2: Consecutive sessions differ in layout

*For any* environment, when `env.reset()` is called twice in succession without providing the same explicit seed, the object position configurations of the two resulting sessions SHALL differ.

**Validates: Requirements 1.3**

---

### Property 3: Collision keeps position and increments count

*For any* environment state where the agent's forward cell is a wall or an occupied cell, issuing `move_forward` SHALL leave the agent's (row, col) position and orientation unchanged and SHALL increment the `collision_count` in the returned `info` dict by exactly 1 compared to its value before the step.

**Validates: Requirements 1.5**

---

### Property 4: POV frame has valid shape and type on every step

*For any* environment state and any valid action, the `obs["image"]` returned by `env.step()` and `env.reset()` SHALL be a numpy array with shape `(H, W, 3)` for positive integers H and W, and dtype `uint8`.

**Validates: Requirements 1.7**

---

### Property 5: CLIP encode_frame produces fixed-size embedding

*For any* numpy array that represents a valid RGB image (any size, uint8 or float, non-null), `encode_frame()` SHALL return a float32 numpy array of shape `(512,)`.

**Validates: Requirements 2.1**

---

### Property 6: compute_similarity output is bounded

*For any* float32 array of shape `(512,)` and any non-empty string, `compute_similarity()` SHALL return a float value in the closed interval `[-1.0, 1.0]`.

**Validates: Requirements 2.3**

---

### Property 7: Vision module raises on invalid inputs

*For any* input to `encode_frame()` that is null, the wrong dtype, or otherwise malformed, a `VisionModuleError` SHALL be raised. *For any* input to `compute_similarity()` where the embedding has shape other than `(512,)` or the text string is empty, a `ValueError` SHALL be raised with a message identifying the invalid argument.

**Validates: Requirements 2.4, 2.6**

---

### Property 8: Goal schema conformance after parsing

*For any* response object returned by the Gemini Pro API — whether valid JSON with extra fields, missing optional fields, or any other structure — `parse_instruction()` SHALL return a dict containing exactly the keys `target`, `attribute`, `context`, and `error`, with no additional keys, and with `target` and `attribute` values at most 50 characters, `context` at most 100 characters.

**Validates: Requirements 3.2, 3.3**

---

### Property 9: Error Goals for all API failure modes

*For any* Gemini Pro API response that is a non-2xx HTTP status code, a timeout, a non-JSON body, or a JSON body missing the `target` field, `parse_instruction()` SHALL return a Goal with `target` set to `null` and an `error` field set to the appropriate error code (`"api_error:{code}"`, `"timeout"`, or `"invalid_response"`). *For any* instruction whose extracted target token is not a member of the environment's `object_types`, the returned Goal SHALL have `error` set to `"unrecognized_target:{token}"`.

**Validates: Requirements 3.4, 3.5, 3.6, 3.9**

---

### Property 10: Instruction sanitization removes Cc code points

*For any* instruction string containing Unicode Cc category code points (U+0000–U+001F or U+007F–U+009F), the string sent to the Gemini Pro API SHALL contain none of those code points, and all other characters SHALL be preserved in order.

**Validates: Requirements 3.7**

---

### Property 11: Policy predict returns valid action for any observation

*For any* float32 numpy array of shape `(576,)`, `predict_action()` SHALL return an integer in `{0, 1, 2, 3}`.

**Validates: Requirements 4.1**

---

### Property 12: Policy serialization round trip

*For any* trained PPO policy, saving it with `policy.save(path)` and loading it with `PPO.load(path)` SHALL produce a policy that returns identical action predictions for any given observation.

**Validates: Requirements 4.4**

---

### Property 13: Goal encoding is consistent and fixed-length

*For any* Goal dict (including Goals with null fields), `encode_goal()` SHALL always return a float32 array of shape `(GOAL_ENCODING_DIM,)`, and calling `encode_goal()` twice with the same Goal dict SHALL return arrays that are element-wise equal.

**Validates: Requirements 4.6**

---

### Property 14: Stop action termination condition

*For any* agent position and target position, when the `stop` action (action index 3) is issued, the session SHALL be marked as success if and only if `manhattan_distance(agent_pos, target_pos) <= 1`, and as failure otherwise.

**Validates: Requirements 4.3**

---

### Property 15: Reward signal correctness

*For any* combination of current agent position, previous agent position, target position, and action:
- A `move_forward` action that decreases Euclidean distance to target SHALL contribute exactly +0.1 to the reward.
- A `move_forward` action that increases Euclidean distance SHALL contribute exactly -0.1.
- A `move_forward` action resulting in a collision (position unchanged) SHALL contribute an additional -0.5.
- A `stop` action with `manhattan_distance <= 1` SHALL return +10.0 terminal reward and `done=True`.
- When `step_count >= step_budget`, the reward SHALL include -5.0 and `done` SHALL be `True`.

**Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6**

---

### Property 16: API returns 404 for unknown session IDs

*For any* UUID string that is not present in the session store, all session endpoints (`/step`, `/status`, `/run`) SHALL return HTTP 404.

**Validates: Requirements 6.5**

---

### Property 17: Terminated session step returns 409

*For any* session that has already reached a terminal state (success, failure, or timeout), calling `POST /session/{id}/step` SHALL return HTTP 409.

**Validates: Requirements 6.2**

---

### Property 18: Request validation returns 422 with field details

*For any* malformed request body sent to any Backend endpoint (missing required fields, wrong field types, values exceeding maximum lengths), the Backend SHALL return HTTP 422 and the response body SHALL contain field-level error details identifying which fields are invalid.

**Validates: Requirements 6.8**

---

### Property 19: Evaluation metrics correctness

*For any* session with a known sequence of actions and agent positions:
- `path_length` SHALL equal the count of `move_forward` actions in the session.
- `steps_to_goal` SHALL equal the total count of all actions taken from session start through the terminal step.
- `collision_count` SHALL equal the count of `move_forward` actions where agent position did not change.
- For successful sessions where `path_length > 0`, `navigation_efficiency` SHALL equal `manhattan_distance(start, target) / path_length`; if either value is zero, efficiency SHALL equal 1.0.
- `success_rate` after N sessions SHALL equal `successful_sessions / N`.

**Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5**

---

### Property 20: Missing config variables all reported at startup

*For any* non-empty subset of required environment variables that are absent at startup, the Backend SHALL log an error message that identifies every missing variable in that subset (not just the first one encountered) and exit with a non-zero status code.

**Validates: Requirements 10.2**

---

## Error Handling

### Vision Module errors

`VisionModuleError` is raised for null input, wrong-type input, or internal CLIP failure. It propagates to the Backend endpoint handler, which catches it and returns HTTP 500 with a generic message. Stack traces are never included in responses.

### Language Module errors

All errors are returned in the Goal dict's `error` field — `parse_instruction()` never raises. The Backend surfaces the error to the Dashboard as a user-visible message in the response body (not HTTP 5xx unless Goal parsing failure is considered unrecoverable, in which case `/session/start` returns HTTP 500 per Requirement 6.1).

### Policy load errors

`load_policy()` raises `RuntimeError` with the file path and failure reason. `api/main.py`'s startup handler catches this, logs it, and exits with code 1. No fallback to retraining.

### Configuration errors

Checked at startup in a single pass. All missing variables collected into a list and logged together before exit. No partial startup.

### API-level error handling

FastAPI exception handler hierarchy:
1. `HTTPException` — passed through as-is (4xx responses)
2. `RequestValidationError` — returns 422 with Pydantic field errors
3. All other exceptions — caught by global handler, returns 500 with message `"Internal server error"`, no stack trace

Session state is never mutated on error — if a step fails mid-execution, the session remains in its pre-step state.

---

## Testing Strategy

### PBT applicability assessment

This feature is a strong candidate for property-based testing. The core modules — `reward.py`, `policy.py` (goal encoding, action prediction), `vision.py` (compute_similarity), `language.py` (parse_instruction, sanitize), `api/session.py` (metrics computation) — are all functions with clear input/output contracts where input variation meaningfully exercises edge cases. The reward function especially benefits from PBT because reward interactions (distance + collision + terminal) must compose correctly across all possible position combinations.

PBT is not applied to:
- Dashboard (UI rendering/interaction, no computable properties)
- Training pipeline (integration, external SB3/file system interaction)
- Gemini API calls (external service, use mocks for unit tests)
- CLIP inference latency (timing test, single execution)

### Property-based testing

**Library:** [Hypothesis](https://hypothesis.readthedocs.io/) (Python, well-maintained, integrates with pytest)

**Configuration:** Each property test runs a minimum of 100 examples (`@settings(max_examples=100)`).

**Tag format:** Each test is tagged with a comment: `# Feature: embodied-navigation-agent, Property {N}: {property_text}`

Property tests to implement (one pytest function per property):

| Property | Module under test | What varies |
|---|---|---|
| P1: Environment layout invariants | `env.py` | grid_size, seed |
| P2: Consecutive sessions differ | `env.py` | seed pairs |
| P3: Collision keeps position | `env.py` | agent pos, obstacle pos |
| P4: POV frame shape on every step | `env.py` | action, env state |
| P5: encode_frame output shape | `vision.py` | image content, size |
| P6: compute_similarity bounded | `vision.py` | embeddings, text strings |
| P7: Vision module raises on invalid | `vision.py` | bad inputs |
| P8: Goal schema conformance | `language.py` | Gemini response shapes |
| P9: Error Goals for API failures | `language.py` | error codes, bad JSON |
| P10: Instruction sanitization | `language.py` | Cc codepoint insertion |
| P11: Policy predict valid action | `policy.py` | observation vectors |
| P12: Policy serialization round trip | `policy.py` | trained policy |
| P13: Goal encoding consistent | `policy.py` | Goal dicts |
| P14: Stop termination condition | `reward.py` + session logic | pos pairs |
| P15: Reward signal correctness | `reward.py` | pos tuples, step counts |
| P16: 404 for unknown sessions | `api/main.py` | UUID strings |
| P17: 409 for terminated sessions | `api/main.py` | terminal sessions |
| P18: 422 for malformed requests | `api/main.py` | bad request bodies |
| P19: Evaluation metrics correctness | `api/session.py` | action sequences |
| P20: Missing config all reported | startup code | env var subsets |

### Unit tests

Example-based tests for:
- Specific reward values at exact position configurations (smoke-check reward table)
- `/session/start` happy path: valid instruction → session ID + base64 PNG
- `/session/{id}/run` timeout path: step budget exhausted → failure response
- Training script: `--force-retrain` flag ignores existing checkpoint
- Dashboard renders correctly with mocked Backend responses (Streamlit `AppTest`)

### Integration tests

- Full episode run: start session → run to completion → metrics correct
- Policy save/load cycle with real SB3 API
- Backend startup with missing env vars → non-zero exit
- Backend startup with corrupt model file → non-zero exit
- `POST /session/{id}/run` completes within 60 seconds for step budget ≤ 200

### Test organization

```
tests/
├── test_env.py          # P1–P4
├── test_vision.py       # P5–P7
├── test_language.py     # P8–P10
├── test_policy.py       # P11–P13
├── test_reward.py       # P14–P15
├── test_api.py          # P16–P18, unit tests for endpoints
├── test_metrics.py      # P19
├── test_config.py       # P20
└── test_integration.py  # full episode, save/load, startup failure modes
```
