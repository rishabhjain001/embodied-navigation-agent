# Embodied Navigation Agent

An embodied navigation agent that follows natural language instructions in a simulated grid environment. It combines CLIP for visual perception, Gemini for instruction parsing, and a PPO policy trained with Stable-Baselines3.

The agent receives a free-text instruction (e.g., "Go to the red chair"), perceives its surroundings through a first-person camera via CLIP, plans a path using a PPO policy, and executes navigation in a MiniGrid environment. A FastAPI backend orchestrates all components; a Streamlit dashboard provides a live view of the agent's point-of-view, step log, and evaluation metrics.

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

---

## Project Structure

```
agent/          # core RL modules
  env.py        # MiniGrid environment wrapper
  vision.py     # CLIP visual encoder
  language.py   # Gemini instruction parser
  policy.py     # SB3 PPO policy wrapper
  reward.py     # stateless reward computation
api/            # FastAPI backend
  main.py       # endpoints and startup
  session.py    # in-memory session store
  models.py     # Pydantic schemas
dashboard/      # Streamlit frontend
  app.py
scripts/
  evaluate.py   # standalone evaluation
train.py        # training entry point
configs/
  default.yaml  # PPO hyperparams and env settings
tests/          # property-based and integration tests
```

---

## Setup

### Requirements

Python 3.10 or 3.11. CLIP installs directly from GitHub and is not on PyPI.

```bash
# 1. Install dependencies (CLIP installs from GitHub, not PyPI)
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env — at minimum set GEMINI_API_KEY and MODEL_PATH

# 3. Train the policy
python train.py --model-path models/policy.zip

# 4. Start the API backend
uvicorn api.main:app --reload --port 8000

# 5. Start the Streamlit dashboard (in a separate terminal)
streamlit run dashboard/app.py
```

### Environment variables

All configurable values live in `.env`. The required keys are:

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Gemini Pro API key — obtain at https://aistudio.google.com/ |
| `MODEL_PATH` | Path to the saved PPO weights file, e.g. `models/policy.zip` |
| `GRID_SIZE` | Grid world side length (minimum 8, default 10) |
| `STEP_BUDGET` | Maximum steps per episode before timeout (default 200) |
| `OBJECT_COUNT` | Objects placed per episode (default 5) |

The backend will log every missing variable and exit with a non-zero status code if any required key is absent.

---

## Known Limitations

- Single-room environment only — no multi-room navigation.
- Exactly 5 objects per episode.
- Single-step instructions only — sequences like "go to the lamp, then the chair" are not supported.
- No persistent memory between sessions — the agent has no recollection of prior episodes.
- CLIP inference runs on CPU — expect roughly 100–300 ms per step depending on hardware.
- The policy is untrained out of the box; run `train.py` before using the backend or the agent will behave randomly.
- Session state is in-memory only — restarting the backend clears all active sessions.

Multi-room navigation, object memory, dynamic obstacles, and multi-step instructions are planned for later phases and are out of scope for this MVP.

---

## Running Tests

```bash
pytest tests/
```

Tests cover reward computation, vision module contracts, language module error handling, policy serialization, environment layout invariants, API endpoint behavior, and evaluation metrics correctness. Property-based tests use [Hypothesis](https://hypothesis.readthedocs.io/).

---

## Citations

Chevalier-Boisvert, M., Willems, L., & Pal, S. (2018). Minimalistic Gridworld Environment for Gymnasium. GitHub. https://github.com/Farama-Foundation/MiniGrid

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., ... & Sutskever, I. (2021). Learning transferable visual models from natural language supervision. *Proceedings of the 38th International Conference on Machine Learning (ICML)*.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv:1707.06347*.
