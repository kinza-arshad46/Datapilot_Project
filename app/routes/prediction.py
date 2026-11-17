"""
Prediction Studio
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, Optional
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import json

from app.routes.auth import get_current_user
from app.ml.preprocessing import DataPilotPreprocessor
from app.database import get_db
from app.ml.explainability import get_feature_importance

router = APIRouter()


class PredictRequest(BaseModel):
    experiment_id: int
    features: Dict[str, Any]


@router.get("/schema/{experiment_id}")
async def get_input_schema(experiment_id: int, current_user: dict = Depends(get_current_user)):
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

    data = joblib.load(row["model_path"])
    feature_names = data.get("feature_names", [])
    target = data.get("target")
    problem_type = data.get("type", "classification")

    # Try to get original column types from preprocessor
    schema = []
    if row["preprocessing_path"]:
        try:
            preproc = DataPilotPreprocessor.load(row["preprocessing_path"])
            for col in preproc.numerical_cols_:
                schema.append({"name": col, "type": "number", "required": True})
            for col in preproc.categorical_cols_:
                schema.append({"name": col, "type": "string", "required": True})
        except Exception:
            for f in feature_names:
                schema.append({"name": f, "type": "number", "required": True})
    else:
        for f in feature_names:
            schema.append({"name": f, "type": "number", "required": True})

    return {
        "experiment_id": experiment_id,
        "algorithm": row["algorithm"],
        "problem_type": problem_type,
        "target": target,
        "schema": schema,
        "feature_names": feature_names,
    }


@router.post("/predict")
async def predict(req: PredictRequest, current_user: dict = Depends(get_current_user)):
    with get_db() as conn:
        c = conn.cursor()
        c.execute(
            """SELECT e.*, p.user_id, p.id as project_id FROM experiments e
               JOIN projects p ON e.project_id = p.id WHERE e.id = ?""",
            (req.experiment_id,)
        )
        row = c.fetchone()
        if not row or row["user_id"] != current_user["user_id"]:
            raise HTTPException(404, "Experiment not found")

    data = joblib.load(row["model_path"])
    model = data["model"]
    problem_type = data.get("type", "classification")
    feature_names = data.get("feature_names", [])

    if not row["preprocessing_path"]:
        raise HTTPException(400, "No preprocessing pipeline found for this model")

    preproc = DataPilotPreprocessor.load(row["preprocessing_path"])

    # Build single-row DataFrame from input
    input_df = pd.DataFrame([req.features])

    try:
        X = preproc.transform(input_df)
    except Exception as e:
        raise HTTPException(400, f"Preprocessing failed: {str(e)}. Check input features.")

    try:
        pred = model.predict(X)
        result = {"prediction": pred[0].item() if hasattr(pred[0], "item") else pred[0]}

        if problem_type == "classification" and hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            classes = getattr(model, "classes_", list(range(len(proba))))
            result["probabilities"] = {
                str(c): round(float(p), 4) for c, p in zip(classes, proba)
            }
            # Decode label if target encoder exists
            if hasattr(preproc, "target_encoder_") and preproc.target_encoder_ is not None:
                try:
                    result["prediction_label"] = preproc.target_encoder_.inverse_transform([int(pred[0])])[0]
                except Exception:
                    result["prediction_label"] = str(result["prediction"])
            else:
                result["prediction_label"] = str(result["prediction"])
        elif problem_type == "regression":
            result["prediction"] = round(float(result["prediction"]), 4)

        # Feature importance for explanation panel
        importance = get_feature_importance(model, feature_names)

        # Save prediction
        with get_db() as conn:
            c = conn.cursor()
            c.execute(
                """INSERT INTO predictions (project_id, experiment_id, input_summary, output, timestamp)
                   VALUES (?, ?, ?, ?, ?)""",
                (row["project_id"], req.experiment_id,
                 json.dumps(req.features), json.dumps(result),
                 datetime.utcnow().isoformat())
            )
            conn.commit()

        return {
            "success": True,
            "result": result,
            "feature_importance": importance[:10],
            "algorithm": row["algorithm"],
        }
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}")
