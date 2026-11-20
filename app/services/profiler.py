"""
Dataset Profiler & Health Score
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import json


def profile_dataset(df: pd.DataFrame, filename: str = "dataset") -> Dict[str, Any]:
    """Generate comprehensive dataset profile and health score."""
    n_rows, n_cols = df.shape
    dtypes = df.dtypes.astype(str).to_dict()

    numerical = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    datetime_cols = []
    for col in df.columns:
        if col in numerical or col in categorical:
            continue
        try:
            pd.to_datetime(df[col], errors="raise")
            datetime_cols.append(col)
        except Exception:
            pass

    # Missing values
    missing = {}
    for col in df.columns:
        miss = int(df[col].isna().sum())
        missing[col] = {
            "count": miss,
            "pct": round(miss / n_rows * 100, 2) if n_rows else 0
        }

    # Duplicates
    n_duplicates = int(df.duplicated().sum())

    # Constant columns
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]

    # High cardinality
    high_card = []
    for c in categorical:
        nuniq = df[c].nunique()
        if nuniq > min(50, n_rows * 0.5):
            high_card.append({"column": c, "unique": int(nuniq)})

    # Candidate targets
    candidates = []
    for c in df.columns:
        nuniq = df[c].nunique()
        if c in categorical or (c in numerical and nuniq <= 20):
            dist = df[c].value_counts(normalize=True).head(10).to_dict()
            candidates.append({
                "column": c,
                "type": "classification" if nuniq <= 20 else "regression",
                "nunique": int(nuniq),
                "distribution": {str(k): round(float(v), 4) for k, v in dist.items()}
            })
        elif c in numerical and nuniq > 20:
            candidates.append({
                "column": c,
                "type": "regression",
                "nunique": int(nuniq),
                "distribution": {}
            })

    # Health score (transparent composite)
    factors = {}
    # Completeness
    total_cells = n_rows * n_cols
    missing_cells = sum(m["count"] for m in missing.values())
    completeness = 1 - (missing_cells / total_cells) if total_cells else 0
    factors["completeness"] = round(completeness * 100, 1)

    # Duplicate free
    dup_free = 1 - (n_duplicates / n_rows) if n_rows else 0
    factors["duplicate_free"] = round(dup_free * 100, 1)

    # Type consistency (no mixed types penalty - simple)
    factors["type_consistency"] = 95.0  # baseline

    # Reasonable size
    size_score = min(100, max(40, (n_rows / 1000) * 20 + (n_cols / 10) * 10))
    factors["size_score"] = round(size_score, 1)

    # Diversity (not too many constants)
    diversity = 1 - (len(constant_cols) / n_cols) if n_cols else 0
    factors["diversity"] = round(diversity * 100, 1)

    health = (
        factors["completeness"] * 0.35 +
        factors["duplicate_free"] * 0.25 +
        factors["type_consistency"] * 0.15 +
        factors["size_score"] * 0.15 +
        factors["diversity"] * 0.10
    )
    health = round(min(100, max(0, health)), 1)

    # Class distribution for first candidate classification target
    class_dist = {}
    for cand in candidates:
        if cand["type"] == "classification" and cand["distribution"]:
            class_dist = cand["distribution"]
            break

    return {
        "filename": filename,
        "rows": n_rows,
        "columns": n_cols,
        "dtypes": dtypes,
        "numerical_features": numerical,
        "categorical_features": categorical,
        "datetime_features": datetime_cols,
        "missing_values": missing,
        "duplicates": n_duplicates,
        "constant_columns": constant_cols,
        "high_cardinality": high_card,
        "candidate_targets": candidates[:8],
        "health_score": health,
        "health_factors": factors,
        "class_distribution": class_dist,
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
    }


def get_basic_stats(df: pd.DataFrame) -> Dict[str, Any]:
    numerical = df.select_dtypes(include=[np.number])
    stats = {}
    if not numerical.empty:
        desc = numerical.describe().to_dict()
        stats["numerical"] = {k: {kk: round(float(vv), 4) if pd.notna(vv) else None for kk, vv in v.items()} for k, v in desc.items()}
    return stats
