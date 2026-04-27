# Meta-Guard

A two-stage Intrusion Detection System Demo.

## Setup
1. Create a `.env` file from `.env.example`.
2. To use the AI Analyst layer, set your `GEMINI_API_KEY` in the `.env` file.
   ```
   GEMINI_API_KEY=your_key_here
   ```
3. Run `start.bat`. IoMT IDS Demo

Meta-Guard is a two-stage intrusion detection demo for IoMT traffic.

- Stage 1 (Edge): Lightweight binary classifier for `normal` vs `suspicious` traffic.
- Stage 2 (Cloud): Open-set embedding analysis for `known attack` vs `unknown attack` (zero-day candidate).

This repository now includes a complete FastAPI backend and Streamlit frontend wired to your model artifacts.

## Project Structure

```text
backend/
  app/
    config.py
    main.py
    models.py
    pipeline.py
    schemas.py
frontend/
  streamlit_app.py
examples/
  sample_inputs.json
models/
  metaguard_level1_reptile.pt.zip
  metaguard_level2_openset.pt.zip
  metaguard_level2_encoder.pkl
  metaguard_scaler.pkl
  metaguard_prototypes.pt.zip
```

## 1. Setup

Recommended Python version: **3.11 or 3.12** for maximum compatibility.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Optional configuration:

```powershell
copy .env.example .env
```

## 2. Run Backend (FastAPI)

```powershell
uvicorn backend.app.main:app --reload --port 8000
```

Available endpoints:

- `GET /health`
- `GET /metadata`
- `POST /predict`

### `/predict` input formats

Single sample as list:

```json
[0.1, 1.2, 0.3, 0.0, 1.0, 0.4, 0.7, 0.9, 1.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.3, 1.4, 0.9, 0.2, 0.1, 0.2, 0.3, 0.2, 0.1, 0.5, 0.4, 0.7, 0.2, 0.1, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
```

Batch input:

```json
{
  "features": [
    [0.1, 1.2, 0.3, 0.0, 1.0, 0.4, 0.7, 0.9, 1.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.3, 1.4, 0.9, 0.2, 0.1, 0.2, 0.3, 0.2, 0.1, 0.5, 0.4, 0.7, 0.2, 0.1, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
    [0.2, 1.5, 0.4, 0.1, 1.3, 0.2, 0.9, 1.0, 1.4, 0.3, 0.5, 0.6, 0.9, 0.4, 0.7, 1.0, 1.2, 1.3, 1.4, 0.2, 0.2, 0.4, 0.5, 0.2, 0.5, 1.0, 1.6, 1.0, 0.1, 0.3, 0.3, 0.2, 0.3, 0.2, 0.6, 0.8, 0.9, 0.2, 0.3, 0.7, 0.6, 0.5, 0.3, 0.3, 0.2]
  ]
}
```

## 3. Run Frontend (Streamlit)

```powershell
streamlit run frontend/streamlit_app.py
```

Use the UI to:

- Enter 45 features manually
- Or upload a CSV (45 numeric columns)
- Run detection and inspect stage-wise results
- Highlight potential zero-day attacks
- Visualize confidence and distance behavior

## 4. Notes for Demo Presentation

- Explain edge/cloud split: edge filters normal traffic quickly, cloud focuses on suspicious traffic.
- Point to open-set distance threshold for zero-day detection logic.
- Use `examples/sample_inputs.json` for quick demo payloads.

Optional: regenerate optimized sample payloads (normal, known attack, unknown attack)

```powershell
python tools/generate_demo_samples.py
```

## 5. Troubleshooting

- If model loading fails, confirm files exist in `models/` with exact names.
- If API returns feature-length error, verify each sample has 45 values.
- If dependency issues appear on very new Python versions, switch to Python 3.11/3.12.
