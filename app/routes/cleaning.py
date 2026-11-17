"""
Data Cleaning endpoints
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Optional, List
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from app.routes.auth import get_current_user
from app.database import get_db
from app.services.profiler import profile_dataset

router = APIRouter()


class CleaningConfig(BaseModel):
    project_id: int
    missing_strategies: Dict[str, str] = {}  # col -> mean/median/most_frequent/drop
    drop_duplicates: bool = True
    drop_columns: List[str] = []
    outlier_action: str = "none"  # none / review / drop
    type_conversions: Dict[str, str] = {}


def load_project_df(project_id: int, user_id: int) -> tuple:
    with get_db() as conn:
        c = conn.cursor()
        c.execute(
            """SELECT d.filepath, d.filename FROM datasets d
               JOIN projects p ON d.project_id = p.id
               WHERE d.project_id = ? AND p.user_id = ?""",
            (project_id, user_id)
        )
        row = c.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Dataset not found")
    path = row["filepath"]
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    return df, path, row["filename"]


@router.post("/apply")
async def apply_cleaning(cfg: CleaningConfig, current_user: dict = Depends(get_current_user)):
    df, path, filename = load_project_df(cfg.project_id, current_user["user_id"])
    original_shape = df.shape

    # Drop columns
    for col in cfg.drop_columns:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Missing value strategies
    for col, strategy in cfg.missing_strategies.items():
        if col not in df.columns:
            continue
        if strategy == "drop":
            df = df.dropna(subset=[col])
        elif strategy == "mean" and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
        elif strategy == "median" and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        elif strategy == "most_frequent":
            mode = df[col].mode()
            if len(mode):
                df[col] = df[col].fillna(mode.iloc[0])
        # else leave as is

    # Global missing fill remaining numerical with median, cat with mode
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                mode = df[col].mode()
                if len(mode):
                    df[col] = df[col].fillna(mode.iloc[0])

    # Duplicates
    if cfg.drop_duplicates:
        df = df.drop_duplicates()

    # Outliers (IQR drop)
    if cfg.outlier_action == "drop":
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower) & (df[col] <= upper)]

    # Type conversions
    for col, dtype in cfg.type_conversions.items():
        if col in df.columns:
            try:
                if dtype == "numeric":
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                elif dtype == "category":
                    df[col] = df[col].astype("category")
                elif dtype == "string":
                    df[col] = df[col].astype(str)
            except Exception:
                pass

    # Save cleaned version
    cleaned_path = Path(path).with_name(Path(path).stem + "_cleaned.csv")
    df.to_csv(cleaned_path, index=False)

    # Update DB
    profile = profile_dataset(df, filename=filename + " (cleaned)")
    with get_db() as conn:
        c = conn.cursor()
        c.execute(
            """UPDATE datasets SET filepath=?, rows=?, columns=?, schema_json=?, health_score=?
               WHERE project_id=?""",
            (
                str(cleaned_path),
                profile["rows"],
                profile["columns"],
                json.dumps(profile["dtypes"]),
                profile["health_score"],
                cfg.project_id,
            )
        )
        conn.commit()

    return {
        "success": True,
        "original_shape": original_shape,
        "new_shape": df.shape,
        "profile": profile,
        "message": "Cleaning applied successfully",
    }


@router.get("/suggestions/{project_id}")
async def cleaning_suggestions(project_id: int, current_user: dict = Depends(get_current_user)):
    df, _, _ = load_project_df(project_id, current_user["user_id"])
    suggestions = {"missing": {}, "duplicates": int(df.duplicated().sum()), "outliers": {}}

    for col in df.columns:
        miss = int(df[col].isna().sum())
        if miss > 0:
            pct = round(miss / len(df) * 100, 2)
            if pd.api.types.is_numeric_dtype(df[col]):
                suggestions["missing"][col] = {"count": miss, "pct": pct, "recommended": "median"}
            else:
                suggestions["missing"][col] = {"count": miss, "pct": pct, "recommended": "most_frequent"}

    # Simple outlier counts
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
        if outliers > 0:
            suggestions["outliers"][col] = int(outliers)

    return suggestions
