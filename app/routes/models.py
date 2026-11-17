"""
Model Lab - Train, Evaluate, Compare
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
import json
import uuid

from app.routes.auth import get_current_user
from app.routes.cleaning import load_project_df
from app.ml.preprocessing import DataPilotPreprocessor
from app.ml.classification import get_classifiers, evaluate_classification
from app.ml.regression import get_regressors, evaluate_regression
from app.ml.clustering import get_clusterers, evaluate_clustering
from app.ml.evaluation import format_metrics
from app.ml.explainability import get_feature_importance, get_permutation_importance, get_shap_values
from app.database import get_db

router = APIRouter()
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class TrainRequest(BaseModel):
    project_id: int
    target: str
    problem_type: str  # classification / regression / clustering
    algorithms: List[str] = []  # empty = all
    test_size: float = 0.2
    random_state: int = 42
    scaler: str = "standard"
    n_clusters: int = 3


@router.post("/train")
async def train_models(req: TrainRequest, current_user: dict = Depends(get_current_user)):
    df, _, _ = load_project_df(req.project_id, current_user["user_id"])

    if req.problem_type not in ("classification", "regression", "clustering"):
        raise HTTPException(400, "Invalid problem_type")

    results = []
    preproc = DataPilotPreprocessor(scaler=req.scaler)

    if req.problem_type == "clustering":
        # No target needed
        X = df.select_dtypes(include=[np.number]).dropna()
        if X.shape[0] < 10 or X.shape[1] < 1:
            raise HTTPException(400, "Need numerical data for clustering")
        from sklearn.preprocessing import StandardScaler
        Xs = StandardScaler().fit_transform(X)
        clusterers = get_clusterers(n_clusters=req.n_clusters, random_state=req.random_state)
        if req.algorithms:
            clusterers = {k: v for k, v in clusterers.items() if k in req.algorithms}

        for name, model in clusterers.items():
            try:
                labels = model.fit_predict(Xs)
                metrics = evaluate_clustering(Xs, labels)
                exp_id = str(uuid.uuid4())[:8]
                model_path = MODELS_DIR / f"{req.project_id}_{exp_id}_{name.replace(' ', '_')}.joblib"
                joblib.dump({"model": model, "type": "clustering"}, model_path)

                with get_db() as conn:
                    c = conn.cursor()
                    c.execute(
                        """INSERT INTO experiments (project_id, algorithm, parameters, metrics, model_path, timestamp)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (req.project_id, name, json.dumps({"n_clusters": req.n_clusters}),
                         json.dumps(metrics), str(model_path), datetime.utcnow().isoformat())
                    )
                    conn.commit()
                    eid = c.lastrowid

                results.append({
                    "experiment_id": eid,
                    "algorithm": name,
                    "metrics": format_metrics(metrics, "clustering"),
                    "status": "success",
                })
            except Exception as e:
                results.append({"algorithm": name, "status": "failed", "error": str(e)})
        return {"results": results, "problem_type": "clustering"}

    # Supervised
    if req.target not in df.columns:
        raise HTTPException(400, f"Target '{req.target}' not found")

    try:
        X_train, X_test, y_train, y_test, feature_names = preproc.fit_transform(
            df, target=req.target, problem_type=req.problem_type,
            test_size=req.test_size, random_state=req.random_state
        )
    except Exception as e:
        raise HTTPException(400, f"Preprocessing failed: {str(e)}")

    if req.problem_type == "classification":
        model_dict = get_classifiers(req.random_state)
    else:
        model_dict = get_regressors(req.random_state)

    if req.algorithms:
        model_dict = {k: v for k, v in model_dict.items() if k in req.algorithms}

    for name, model in model_dict.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if req.problem_type == "classification":
                y_proba = None
                if hasattr(model, "predict_proba"):
                    try:
                        y_proba = model.predict_proba(X_test)
                    except Exception:
                        pass
                metrics = evaluate_classification(y_test, y_pred, y_proba)
            else:
                metrics = evaluate_regression(y_test, y_pred)

            # Importance
            importance = get_feature_importance(model, feature_names)

            exp_id = str(uuid.uuid4())[:8]
            model_path = MODELS_DIR / f"{req.project_id}_{exp_id}_{name.replace(' ', '_')}.joblib"
            preproc_path = MODELS_DIR / f"{req.project_id}_{exp_id}_preproc.joblib"
            joblib.dump({"model": model, "type": req.problem_type, "feature_names": feature_names,
                         "target": req.target}, model_path)
            preproc.save(str(preproc_path))

            with get_db() as conn:
                c = conn.cursor()
                c.execute(
                    """INSERT INTO experiments
                       (project_id, algorithm, parameters, metrics, model_path, preprocessing_path, timestamp)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (req.project_id, name, json.dumps({"test_size": req.test_size}),
                     json.dumps({k: v for k, v in metrics.items() if k not in ("actual", "predicted", "report")}),
                     str(model_path), str(preproc_path), datetime.utcnow().isoformat())
                )
                conn.commit()
                eid = c.lastrowid

            results.append({
                "experiment_id": eid,
                "algorithm": name,
                "metrics": format_metrics(metrics, req.problem_type),
                "feature_importance": importance,
                "status": "success",
            })
        except Exception as e:
            results.append({"algorithm": name, "status": "failed", "error": str(e)})

    # Update project problem type
    with get_db() as conn:
        c = conn.cursor()
        c.execute("UPDATE projects SET problem_type=? WHERE id=?", (req.problem_type, req.project_id))
        conn.commit()

    return {
        "results": results,
        "problem_type": req.problem_type,
        "n_features": len(feature_names) if 'feature_names' in dir() else 0,
        "train_size": len(y_train) if 'y_train' in dir() else 0,
        "test_size": len(y_test) if 'y_test' in dir() else 0,
    }


@router.get("/experiments/{project_id}")
async def list_experiments(project_id: int, current_user: dict = Depends(get_current_user)):
    with get_db() as conn:
        c = conn.cursor()
        c.execute(
            """SELECT e.* FROM experiments e
               JOIN projects p ON e.project_id = p.id
               WHERE e.project_id = ? AND p.user_id = ?
               ORDER BY e.timestamp DESC""",
            (project_id, current_user["user_id"])
        )
        rows = c.fetchall()
    return [
        {
            "id": r["id"],
            "algorithm": r["algorithm"],
            "metrics": json.loads(r["metrics"] or "{}"),
            "timestamp": r["timestamp"],
            "model_path": r["model_path"],
        }
        for r in rows
    ]


@router.get("/explain/{experiment_id}")
async def explain_model(experiment_id: int, current_user: dict = Depends(get_current_user)):
    with get_db() as conn:
        c = conn.cursor()
        c.execute(
            """SELECT e.*, p.user_id FROM experiments e
               JOIN projects p ON e.project_id = p.id WHERE e.id = ?""",
            (experiment_id,)
        )
        row = c.fetchone()
        if not row or row["user_id"] != current_user["user_id"]:
            raise HTTPException(404, "Experiment not found")

    try:
        data = joblib.load(row["model_path"])
        model = data["model"]
        feature_names = data.get("feature_names", [])
        importance = get_feature_importance(model, feature_names)

        # Optional permutation / SHAP if we can load data
        result = {
            "feature_importance": importance,
            "algorithm": row["algorithm"],
            "metrics": json.loads(row["metrics"] or "{}"),
        }

        # Try SHAP if tree-based and small
        if hasattr(model, "feature_importances_") and row.get("preprocessing_path"):
            try:
                preproc = DataPilotPreprocessor.load(row["preprocessing_path"])
                # We don't have original X here easily; skip full SHAP for speed
                result["shap"] = {"note": "Enable on Prediction Studio for individual explanations"}
            except Exception:
                pass

        return result
    except Exception as e:
        raise HTTPException(500, f"Failed to load model: {str(e)}")
