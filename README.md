# Fake vs Real News Classification
**ML Engineer Take-Home Assignment**

---

## Project Structure

```
MLE_case_study/
├── data/
│   └── raw/                         ← place Fake.csv and True.csv here
├── notebooks/
│   ├── track1_classical_ml.ipynb    ← Track 1: full EDA + TF-IDF + LinearSVC
│   └── track2_llm.ipynb             ← Track 2: hybrid LLM pipeline
├── src/
│   ├── preprocessing.py             ← data loading, cleaning, splitting
│   ├── train.py                     ← training script (run once)
│   └── llm_inference.py             ← GPT-4o-mini structured output + hybrid router
├── app/
│   ├── api.py                       ← FastAPI REST API (/predict, /predict/batch)
│   └── dashboard.py                 ← Streamlit 4-tab interactive dashboard
├── outputs/
│   ├── figures/                     ← all saved plots (auto-created on train)
│   ├── metrics/                     ← CSV metric reports (auto-created on train)
│   ├── models/                      ← saved joblib pipelines (auto-created on train)
│   └── predictions/                 ← test-set predictions
├── docs/
│   └── assignment_answers.docx
├── presentation/
│   └── fake_news_classifier.pptx
├── .env                             ← API key (not committed)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Note on Project Structure

The primary analysis, including exploratory data analysis (EDA) and model evaluation, is presented in the Jupyter notebooks (`.ipynb`) to ensure clarity and reproducibility. All assignment questions are answered in `assignment_answers.docx`.

Alongside this, a production-oriented Python version of the project has been implemented (`src/` + `app/`), supporting model training, inference, API deployment, and an interactive dashboard.

## Setup

```bash
# 1. Install all dependencies
pip install -r requirements.txt

# 2. Place data files
#    Either copy to data/raw/ (preferred) or leave in project root
cp Fake.csv True.csv data/raw/

# 3. Add your OpenAI API key (for Track 2 only)
#    Edit .env and replace the placeholder:
OPENAI_API_KEY=your_openai_api_key_here
```

---

## Run Order (Important)

### Step 1 — Train models (required before API or dashboard)

```bash
python src/train.py
```

This produces:
- `outputs/models/lr_pipeline.joblib`
- `outputs/models/svc_pipeline.joblib`
- `outputs/metrics/track1_metrics.csv`
- `outputs/figures/` — all evaluation plots

---

### Step 2a — Run the Streamlit Dashboard (recommended for demo)

```bash
streamlit run app/dashboard.py
```

Opens at `http://localhost:8501` with 4 tabs:
- **Live Demo** — paste any article, get instant prediction + confidence gauge
- **EDA & Insights** — dataset visualizations and key findings
- **Model Performance** — metrics table, confusion matrix, ROC/PR curves, robustness
- **Business Case** — decision matrix, deployment guide, improvement roadmap

To run without OpenAI (no LLM tab):
```bash
NO_LLM=1 streamlit run app/dashboard.py
```

---

### Step 2b — Run the FastAPI REST API

```bash
uvicorn app.api:app --reload
```

Docs at `http://localhost:8000/docs`

**Classify a single article:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "WASHINGTON (Reuters) - The Senate passed the bill...", "model": "svc"}'
```

**With LLM escalation:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "BREAKING: Deep state exposed!!!", "model": "svc", "use_llm": true}'
```

**Endpoints:**
| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health + model status |
| GET | `/health` | Health check |
| POST | `/predict` | Classify a single article |
| POST | `/predict/batch` | Classify up to 50 articles |
| GET | `/metrics` | Last saved Track 1 metrics |

---

### Step 3 — Run the Notebooks (for full EDA + evaluation walkthrough)

```bash
jupyter notebook notebooks/track1_classical_ml.ipynb
# Run all cells — Kernel → Restart & Run All
```

```bash
jupyter notebook notebooks/track2_llm.ipynb
# Requires OPENAI_API_KEY in .env
# Or set USE_LLM = False in the Config cell for baseline-only mode
```

---

## Approach Summary

| | Track 1 | Track 2 |
|---|---|---|
| Method | TF-IDF (50K, bigrams) + LinearSVC | LR baseline + GPT-4o-mini |
| Imbalance | `class_weight='balanced'` | Same |
| Inference latency | < 1 ms / article | < 1 ms (baseline) / 1–3 s (LLM) |
| Cost | ~$0 | ~$0.00022/LLM call (~4% routing) |
| Deployment | Single `joblib` file | joblib + OpenAI API |
| Interpretability | Low | High — textual reasons per article |

---

## Key Design Decisions

| Decision | Reason |
|---|---|
| Dedup before split | Prevents test contamination |
| TF-IDF fit on train only | Prevents leakage via vocabulary |
| Stratified 80/20 split | Preserves 27/73 class ratio |
| `class_weight='balanced'` over SMOTE | SMOTE produces unrealistic synthetic samples in sparse TF-IDF space |
| `subject` column dropped | Near-perfect label proxy — would cause severe leakage |
| CalibratedClassifierCV on SVC | LinearSVC has no native `predict_proba`; Platt scaling adds calibrated probabilities needed for hybrid routing |

---

## Secrets

- `OPENAI_API_KEY` must be set in `.env` for Track 2 LLM calls.
- The key is never committed (`.gitignore` excludes `.env`).
- All Track 1 components and the API/dashboard run without any API key.
- To disable LLM entirely: set `NO_LLM=1` or `USE_LLM = False`.

---

## Regenerate Notebooks

```bash
python generate_notebooks.py
```

Rewrites both `.ipynb` files from source. Useful if you want to reset to the clean state.
