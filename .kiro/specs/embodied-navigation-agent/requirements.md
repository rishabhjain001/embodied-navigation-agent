# Requirements Document

## Introduction

The Embodied Navigation Agent is an AI system that navigates a simulated indoor environment by interpreting natural language instructions and using visual perception. An agent receives a free-text instruction (e.g., "Go to the red chair"), perceives its surroundings through a first-person camera processed by CLIP, plans a path via a PPO-trained policy, and executes the navigation until it reaches the target or exhausts its step budget.

The MVP scope is a single-room MiniGrid environment containing 5 objects, where the agent successfully reaches a target object identified from a natural language instruction. A Streamlit dashboard displays the agent's live point-of-view, step log, and evaluation metrics. Phase 2 (multi-room, object memory, dynamic obstacles) and Phase 3 (multi-step instructions) are out of scope for MVP.

---

## Glossary

- **Agent**: The AI entity that moves through the simulated environment and executes navigation instructions.
- **Environment**: A MiniGrid-based single-room grid world populated with objects.
- **Instruction**: A natural language string provided by the user describing a navigation goal (e.g., "Go to the red chair").
- **Vision_Module**: The CLIP-based component (`openai/clip-vit-base-patch32`) that processes first-person camera frames and extracts object and attribute embeddings.
- **Language_Module**: The Gemini Pro-based component that parses an Instruction into a structured Goal.
- **Goal**: A structured representation of the navigation target, e.g., `{target: "chair", attribute: "red", context: "near table"}`.
- **Navigation_Policy**: A PPO-trained reinforcement learning policy (Stable-Baselines3) that selects the next action given the current observation.
- **Observation**: The combined representation fed to the Navigation_Policy, consisting of the Vision_Module embedding and the Goal encoding.
- **Action**: One of four discrete choices available to the Agent: `move_forward`, `turn_left`, `turn_right`, `stop`.
- **Reward_System**: The component that computes scalar reward signals after each Action.
- **Session**: A single navigation episode from instruction receipt to terminal state (success, collision limit, or step budget exhaustion).
- **Dashboard**: The Streamlit-based frontend displaying live agent POV, step log, and evaluation metrics.
- **Backend**: The FastAPI service that hosts the Agent, Environment, Vision_Module, Language_Module, Navigation_Policy, and Reward_System.
- **Step_Log**: An ordered list of per-step records including action taken, reward received, and agent position.
- **Evaluation_Metrics**: Aggregate statistics computed over a Session or set of Sessions: success rate, path length, steps to goal, collision count, and navigation efficiency.
- **Navigation_Efficiency**: The ratio of the optimal path length to the actual path length taken by the Agent; higher is better.
- **Step_Budget**: The maximum number of steps allowed per Session (MVP default: 200).
- **POV_Frame**: A rendered RGB pixel image representing the Agent's current first-person partial view of the Environment, produced by MiniGrid's `RGBImgPartialObsWrapper`.

---

## Requirements

### Requirement 1: Environment Setup

**User Story:** As a developer, I want a MiniGrid single-room environment with at least 5 object types, so that the Agent has a meaningful scene to navigate.

#### Acceptance Criteria

1. THE Environment SHALL initialize a single rectangular room on a grid of at least 8×8 cells and at most 32×32 cells.
2. WHEN a new Session begins, THE Environment SHALL place exactly 5 objects in the room, each at a distinct grid cell, with each object having a distinct combination of type and color attribute, such that no two objects share the same grid cell.
3. WHEN a new Session begins, THE Environment SHALL assign object positions such that no two consecutive Sessions have identical object position configurations.
4. THE Environment SHALL expose an `object_types` set containing at least 5 distinct object types (e.g., chair, table, lamp, door, box).
5. WHEN the Agent attempts to move into a wall or occupied cell, THE Environment SHALL keep the Agent's position unchanged and increment the `collision_count` field in the step observation by 1.
6. WHEN a new Session begins, THE Environment SHALL spawn the Agent at a randomly selected interior cell that is not occupied by any object, with a randomly selected orientation from the 4 cardinal directions (north, east, south, west).
7. WHEN a step is taken, THE Environment SHALL render a POV_Frame by wrapping the MiniGrid environment with `RGBImgPartialObsWrapper`, producing a full-color RGB pixel image of the Agent's partial forward-facing view, and return it as part of the step observation.

---

### Requirement 2: Vision Module

**User Story:** As a developer, I want CLIP to process the Agent's first-person frames, so that the system can identify objects and attributes in the scene without manual feature engineering.

#### Acceptance Criteria

1. THE Vision_Module SHALL accept a POV_Frame as a RGB image of size 224×224 pixels as input and return a 512-dimensional embedding vector using the `openai/clip-vit-base-patch32` model.
2. WHEN a POV_Frame is provided on a standard x86-64 CPU (single core, no GPU), THE Vision_Module SHALL return the embedding within 500ms wall-clock time.
3. THE Vision_Module SHALL expose a `compute_similarity(frame_embedding, text_description)` function that accepts a 512-dimensional frame embedding vector and a non-empty text string, and returns a scalar cosine similarity score in the range [−1.0, 1.0].
4. IF the `compute_similarity` function receives a frame embedding of incorrect dimension or a text description that is an empty string, THEN THE Vision_Module SHALL raise a `ValueError` with a message describing which argument is invalid.
5. THE Vision_Module SHALL load model weights once at Backend startup and reuse them across all Sessions without reloading.
6. IF the Vision_Module fails to process a POV_Frame (e.g., due to an invalid image format, null input, or internal model error), THEN THE Vision_Module SHALL raise a `VisionModuleError` exception with a message identifying the failure cause and propagate it to the Backend.

---

### Requirement 3: Language Module

**User Story:** As a developer, I want Gemini Pro to parse natural language instructions into structured Goals, so that the Navigation_Policy has a consistent, machine-readable target representation.

#### Acceptance Criteria

1. WHEN an Instruction is received, THE Language_Module SHALL call the Gemini Pro API and return a Goal within 3 seconds wall-clock time from receipt of the Instruction.
2. THE Language_Module SHALL parse the Goal into a JSON object containing exactly the fields `target` (string of at most 50 characters or null), `attribute` (string of at most 50 characters or null), and optionally `context` (string of at most 100 characters); any extra fields returned by Gemini SHALL be discarded.
3. WHERE a spatial context is present in the Instruction as a prepositional phrase referencing a named landmark object (e.g., "near the lamp", "next to the table"), THE Language_Module SHALL populate the `context` field in the Goal JSON with that phrase.
4. IF the Gemini Pro API returns a non-2xx HTTP response, THEN THE Language_Module SHALL return a Goal JSON with `target` set to `null` and an `error` field set to `"api_error"` plus the HTTP status code; THE Backend SHALL surface this error to the Dashboard as a user-visible message without raising an unhandled exception.
5. IF the Gemini Pro API does not respond within 3 seconds, THEN THE Language_Module SHALL cancel the request and return a Goal JSON with `target` set to `null` and an `error` field set to `"timeout"`; THE Backend SHALL surface this error to the Dashboard as a user-visible message without raising an unhandled exception.
6. IF the Instruction does not identify a target whose type is a member of the Environment's `object_types` set, THEN THE Language_Module SHALL return a Goal with `target` set to `null` and an `error` field set to `"unrecognized_target"` plus the unrecognized token.
7. THE Language_Module SHALL sanitize Instruction strings by removing all Unicode Cc category code points (U+0000–U+001F and U+007F–U+009F) before sending to the Gemini Pro API.
8. THE Language_Module SHALL format the Gemini Pro prompt using a fixed system prompt template that instructs the model to respond only with valid JSON matching the Goal schema.
9. WHEN the Gemini Pro API returns a response, THE Language_Module SHALL validate that the response body is valid JSON containing a `target` field before returning it; IF the response is not valid JSON or is missing the `target` field, THEN THE Language_Module SHALL return a Goal with `target` set to `null` and `error` set to `"invalid_response"`.

---

### Requirement 4: Navigation Policy

**User Story:** As a developer, I want a PPO-trained policy that selects actions based on visual and language observations, so that the Agent can learn to navigate toward targets efficiently.

#### Acceptance Criteria

1. THE Navigation_Policy SHALL accept an Observation (concatenation of the Vision_Module embedding and the Goal encoding) and return a single Action from `{move_forward, turn_left, turn_right, stop}`.
2. THE Navigation_Policy SHALL be trained using the Stable-Baselines3 PPO implementation with the MlpPolicy on the MVP Environment.
3. WHEN the Navigation_Policy selects the `stop` Action, THE Session SHALL terminate and be evaluated as a success if the Agent is within 1 grid cell of the target object, or as a failure otherwise.
4. THE Navigation_Policy SHALL be serializable to and deserializable from a file, allowing weights to be saved after training and loaded at Backend startup.
5. WHEN a saved policy file is present at the configured path, THE Backend SHALL load the Navigation_Policy from that file; IF the file cannot be read, is corrupt, or is version-incompatible, THEN THE Backend SHALL log a descriptive error identifying the file path and failure reason and exit with a non-zero status code.
6. THE Navigation_Policy encoding of the Goal SHALL be a fixed-length vector of consistent dimensionality between training and inference, sufficient to distinguish different Goal values.
7. WHEN `train.py` is invoked, THE Navigation_Policy SHALL stop training after the configured maximum number of timesteps and persist the resulting weights to the configured file path.

---

### Requirement 5: Reward System

**User Story:** As a developer, I want a reward function that guides the Agent toward the target efficiently, so that the PPO policy learns meaningful navigation behavior.

#### Acceptance Criteria

1. WHEN the Agent executes a `move_forward` Action and the Euclidean grid distance to the target object strictly decreases compared to the previous step, THE Reward_System SHALL return a reward of +0.1.
2. WHEN the Agent executes a `move_forward` Action and the Euclidean grid distance to the target object strictly increases compared to the previous step, THE Reward_System SHALL return a penalty of −0.1.
3. WHEN the Agent executes a `move_forward` Action and the Euclidean grid distance to the target object is unchanged (e.g., collision blocked movement), THE Reward_System SHALL return a reward of 0.0 for that distance component.
4. WHEN the Agent's attempted `move_forward` Action is blocked by a wall or occupied cell, THE Reward_System SHALL return an additional penalty of −0.5.
5. WHEN the Agent reaches within 1 grid cell of the target object and executes the `stop` Action, THE Reward_System SHALL return a terminal reward of +10.0.
6. WHEN the Agent has taken a number of steps greater than or equal to Step_Budget without reaching the target, THE Reward_System SHALL return a terminal penalty of −5.0 and THE Session SHALL terminate.
7. THE Reward_System SHALL compute distance-based and collision rewards using only the Environment's ground-truth object positions and the Agent's current position.
8. THE Reward_System SHALL NOT use Vision_Module or Language_Module outputs during reward computation.

---

### Requirement 6: Backend API

**User Story:** As a developer, I want a FastAPI backend that orchestrates all components, so that the Dashboard and any external client can control the Agent through a well-defined HTTP interface.

#### Acceptance Criteria

1. THE Backend SHALL expose a `POST /session/start` endpoint that accepts an Instruction string of at most 500 characters, initializes the Environment, parses the Goal, and returns a session ID (UUID) and the initial POV_Frame as a base64-encoded PNG; IF Environment initialization or Goal parsing fails, THE Backend SHALL return HTTP 500 with a sanitized error message.
2. WHEN `POST /session/{session_id}/step` is called on a Session that has already reached a terminal state (success or failure), THE Backend SHALL return HTTP 409 with a message indicating the Session is complete.
3. THE Backend SHALL expose a `GET /session/{session_id}/status` endpoint that returns the current Session state (active, success, failure), the Step_Log, and Evaluation_Metrics.
4. THE Backend SHALL expose a `POST /session/{session_id}/run` endpoint that runs the Session to completion (up to Step_Budget steps) and returns the final Session state and Evaluation_Metrics; this endpoint SHALL complete or return a timeout error within 60 seconds.
5. IF a session ID that does not exist is provided to any session endpoint, THEN THE Backend SHALL return HTTP 404 with a message identifying the unknown session ID.
6. THE Backend SHALL return all responses in JSON format with Content-Type `application/json`, except POV_Frame images which SHALL be base64-encoded strings within JSON payloads.
7. IF an unhandled exception occurs during request processing, THE Backend SHALL return HTTP 500 with an error message that does not include Python stack traces or internal file paths; client errors (invalid session IDs, malformed requests) SHALL return appropriate 4xx status codes.
8. THE Backend SHALL validate all incoming request bodies against defined Pydantic models and return HTTP 422 with field-level error details for invalid inputs.

---

### Requirement 7: Streamlit Dashboard

**User Story:** As a user, I want an interactive Streamlit dashboard, so that I can issue navigation instructions and observe the Agent's behavior and performance in real time.

#### Acceptance Criteria

1. THE Dashboard SHALL display a text input field where the user can type an Instruction of at most 500 characters and submit it to start a new Session.
2. WHEN a Session is active and the Backend returns a step response, THE Dashboard SHALL update and display the current POV_Frame within 1 second of receiving the response.
3. WHEN a Session is active and the Backend returns a step response, THE Dashboard SHALL update and display the Step_Log as a scrollable table showing step number, action, reward, and Agent grid position.
4. WHEN a Session is active, THE Dashboard SHALL display an Evaluation_Metrics panel showing success rate, path length, steps to goal, collision count, and navigation efficiency sourced from the most recent Backend response.
5. WHEN the user clicks the "Run to Completion" button and a Session is active, THE Dashboard SHALL call `POST /session/{session_id}/run` and update the POV_Frame, Step_Log, and Evaluation_Metrics panels with the final state upon receiving the response.
6. WHEN the user clicks the "Step" button and a Session is active, THE Dashboard SHALL call `POST /session/{session_id}/step` once and update all display panels with the response.
7. THE Dashboard SHALL disable the "Step" and "Run to Completion" buttons when no Session is active (before the first Session or after a Session ends).
8. WHILE the Dashboard is awaiting a response from the Backend, THE Dashboard SHALL display a loading indicator and disable the "Step" and "Run to Completion" buttons to prevent concurrent requests.
9. WHEN a Session ends (success or failure), THE Dashboard SHALL display a status banner above the control buttons indicating success, failure due to step budget exhaustion, or failure due to incorrect stop, before any further interaction is possible.
10. IF the Backend returns an error response (any 4xx or 5xx status code), THE Dashboard SHALL display the error message in a visible error area and remain interactive for the user to submit a new Instruction.

---

### Requirement 8: Evaluation Metrics Computation

**User Story:** As a researcher, I want accurate evaluation metrics computed per session, so that I can benchmark the Agent's navigation performance.

#### Acceptance Criteria

1. THE Backend SHALL compute path length as the total number of `move_forward` actions executed in a Session.
2. THE Backend SHALL compute steps to goal as the total number of actions (all types) taken from Session start to the terminal state, where terminal state is defined as the step on which the Agent executes `stop` or the step on which the Step_Budget is exhausted.
3. THE Backend SHALL compute collision count as the total number of `move_forward` actions in a Session where the Agent's position was unchanged due to a wall or occupied-cell collision.
4. WHEN a Session ends in success, THE Backend SHALL compute navigation efficiency as the ratio of the Manhattan distance between the Agent's start position and the target object's position to the actual path length; if path length is zero or Manhattan distance is zero, efficiency SHALL be reported as 1.0.
5. WHEN a Session reaches its terminal state, THE Backend SHALL update the success rate for the current experiment as the number of successful Sessions divided by the total number of Sessions completed in that experiment.
6. THE Backend SHALL include all Evaluation_Metrics in the response body of `GET /session/{session_id}/status` and `POST /session/{session_id}/run`; for metrics that cannot yet be computed (e.g., navigation efficiency for an in-progress Session), the value SHALL be `null`.

---

### Requirement 9: Model Persistence and Training Pipeline

**User Story:** As a developer, I want a training script and model persistence, so that the Navigation_Policy can be trained offline and loaded for inference without retraining on every startup.

#### Acceptance Criteria

1. THE Backend SHALL include a standalone `train.py` script that trains the Navigation_Policy for a configurable number of timesteps and saves the resulting model weights to a configurable file path.
2. WHEN training completes, THE train.py script SHALL log the final mean reward, mean episode length, and episode success rate computed over the last 100 episodes to standard output.
3. THE Navigation_Policy file format SHALL be compatible with Stable-Baselines3's `model.save()` and `model.load()` methods.
4. WHEN `train.py` is invoked without a pre-existing model file at the configured path, THE train.py script SHALL initialize the Navigation_Policy from scratch and begin training.
5. IF Navigation_Policy initialization fails (due to insufficient memory, invalid configuration, or other error), THEN THE train.py script SHALL log a descriptive error message identifying the cause and exit with a non-zero status code without beginning training.
6. WHEN `train.py` is invoked with a pre-existing model file at the configured path, THE train.py script SHALL load the existing weights and continue training from the saved checkpoint.
7. WHEN `train.py` is invoked with a `--force-retrain` flag, THE train.py script SHALL ignore any pre-existing model file and initialize the Navigation_Policy from scratch.
8. IF the Navigation_Policy file cannot be saved after training (due to I/O error or insufficient disk space), THEN THE train.py script SHALL log a descriptive error identifying the file path and failure reason and exit with a non-zero status code.
9. IF the Navigation_Policy file cannot be loaded (due to corruption, version incompatibility, or I/O error), THEN THE Backend SHALL log a descriptive error identifying the file path and failure reason and exit with a non-zero status code rather than falling back to training from scratch.

---

### Requirement 10: Configuration and Environment Variables

**User Story:** As a developer, I want all tuneable parameters in a single configuration layer, so that I can adjust the system without modifying source code.

#### Acceptance Criteria

1. THE Backend SHALL read all configurable parameters (Step_Budget, model file path, Gemini API key, grid size, object count, reward values) from environment variables or a `.env` file loaded at startup.
2. WHEN any required environment variable is missing at startup, THE Backend SHALL log a descriptive error message identifying every missing variable and exit with a non-zero status code immediately, even if other configuration values loaded successfully.
3. WHILE the Backend has successfully loaded its configuration, THE Backend SHALL expose a `GET /config` endpoint that returns the current non-secret configuration values (Step_Budget, grid size, object count, model path) as a JSON object.
4. IF the Backend has not successfully loaded its configuration, THEN `GET /config` SHALL return HTTP 503 with a message indicating configuration loading failed.
5. THE Backend SHALL never include secret values (API keys, tokens) in any HTTP response or log output.
