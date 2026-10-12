# DataPilot AI

**Upload. Analyze. Train. Predict. Understand.**

Complete end-to-end Machine Learning platform — a reusable web application for dataset profiling, data preparation, EDA, model training, comparison, prediction, explainability, dashboards and automated reporting.

## Features

- Secure authentication (JWT + bcrypt)
- CSV / Excel upload with intelligent profiling & health score
- Data cleaning (missing values, duplicates, outliers)
- Interactive EDA Studio (Plotly charts)
- Model Lab: Classification, Regression, Clustering
  - Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM (optional)
  - Linear / Tree / Forest / XGBoost regressors
  - K-Means, DBSCAN
- Side-by-side model comparison with real metrics
- Prediction Studio with dynamic forms & probabilities
- Feature importance / explainability
- Interactive Dashboard
- Automated PDF report generation
- Domain modes (Business, Finance, Healthcare, etc.)
- Docker-ready

## Design

Navy + Blue + Purple glass/card interface with floating ambient effects, responsive layout, professional dashboard feel.

## Quick Start (Local)

```bash
# Clone / extract
cd DataPilot-AI

# Create virtualenv (recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install
pip install -r requirements.txt

# Run
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000

**Demo credentials:** `demo` / `demo123`

## Docker

```bash
docker build -t datapilot-ai .
docker run -p 8000:8000 datapilot-ai
```

## Project Structure

```
DataPilot-AI/
├── app/
│   ├── main.py
│   ├── database.py
│   ├── routes/          # auth, upload, cleaning, eda, models, prediction, reports, dashboard
│   ├── ml/              # preprocessing, classification, regression, clustering, evaluation, explainability
│   ├── services/        # profiler, cleaner, insights
│   ├── templates/       # HTML pages
│   └── static/          # CSS, JS
├── data/                # uploads & SQLite
├── models/              # trained artifacts
├── reports/             # PDF reports
├── requirements.txt
├── Dockerfile
└── README.md
```

## Workflow

Login → Home → Upload Dataset → Intelligent Profile → Data Cleaning → EDA Studio → Model Lab → Model Comparison → Prediction Studio → Explainability → Dashboard → Report

## Notes

- Free hosting (e.g. Render) may sleep and has limited memory — keep datasets moderate.
- SHAP is optional and runs only for manageable sizes.
- Passwords are never stored or displayed in plaintext.
- All major features are real working workflows with validation and error handling.

## License

For educational / portfolio use.
