"""
EDA Studio - Interactive charts data
"""
from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json

from app.routes.auth import get_current_user
from app.routes.cleaning import load_project_df
from app.services.profiler import profile_dataset

router = APIRouter()


def fig_to_json(fig):
    return json.loads(fig.to_json())


@router.get("/overview/{project_id}")
async def eda_overview(project_id: int, current_user: dict = Depends(get_current_user)):
    df, _, filename = load_project_df(project_id, current_user["user_id"])
    profile = profile_dataset(df, filename)
    return {
        "profile": profile,
        "numerical": profile["numerical_features"],
        "categorical": profile["categorical_features"],
    }


@router.get("/distribution/{project_id}")
async def numerical_distribution(
    project_id: int,
    column: str = Query(...),
    current_user: dict = Depends(get_current_user),
):
    df, _, _ = load_project_df(project_id, current_user["user_id"])
    if column not in df.columns:
        raise HTTPException(404, "Column not found")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise HTTPException(400, "Column is not numerical")

    fig = px.histogram(df, x=column, nbins=40, title=f"Distribution of {column}",
                       color_discrete_sequence=["#1677FF"])
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7,26,61,0.6)",
        font=dict(color="#E8F0FF"),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    box = px.box(df, y=column, title=f"Box Plot — {column}",
                 color_discrete_sequence=["#7257FF"])
    box.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7,26,61,0.6)",
        font=dict(color="#E8F0FF"),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return {"histogram": fig_to_json(fig), "boxplot": fig_to_json(box)}


@router.get("/categorical/{project_id}")
async def categorical_freq(
    project_id: int,
    column: str = Query(...),
    current_user: dict = Depends(get_current_user),
):
    df, _, _ = load_project_df(project_id, current_user["user_id"])
    if column not in df.columns:
        raise HTTPException(404, "Column not found")
    counts = df[column].value_counts().head(20)
    fig = px.bar(
        x=counts.index.astype(str), y=counts.values,
        title=f"Frequency — {column}",
        labels={"x": column, "y": "Count"},
        color_discrete_sequence=["#19C6FF"],
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7,26,61,0.6)",
        font=dict(color="#E8F0FF"),
        margin=dict(l=40, r=20, t=50, b=40),
        xaxis_tickangle=-30,
    )
    return {"chart": fig_to_json(fig)}


@router.get("/correlation/{project_id}")
async def correlation_heatmap(project_id: int, current_user: dict = Depends(get_current_user)):
    df, _, _ = load_project_df(project_id, current_user["user_id"])
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        raise HTTPException(400, "Need at least 2 numerical columns")
    corr = num.corr()
    fig = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        color_continuous_scale=["#041127", "#1677FF", "#19C6FF", "#7257FF"],
        title="Correlation Heatmap",
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E8F0FF"),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return {"chart": fig_to_json(fig)}


@router.get("/scatter/{project_id}")
async def scatter(
    project_id: int,
    x: str = Query(...),
    y: str = Query(...),
    color: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
):
    df, _, _ = load_project_df(project_id, current_user["user_id"])
    for c in [x, y]:
        if c not in df.columns:
            raise HTTPException(404, f"Column {c} not found")
    fig = px.scatter(
        df, x=x, y=y, color=color if color and color in df.columns else None,
        title=f"{y} vs {x}",
        color_continuous_scale=["#1677FF", "#7257FF"],
        opacity=0.7,
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7,26,61,0.6)",
        font=dict(color="#E8F0FF"),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return {"chart": fig_to_json(fig)}


@router.get("/target/{project_id}")
async def target_analysis(
    project_id: int,
    target: str = Query(...),
    current_user: dict = Depends(get_current_user),
):
    df, _, _ = load_project_df(project_id, current_user["user_id"])
    if target not in df.columns:
        raise HTTPException(404, "Target not found")

    result = {}
    if pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique() > 15:
        # Regression target dist
        fig = px.histogram(df, x=target, nbins=40, title=f"Target Distribution — {target}",
                           color_discrete_sequence=["#7257FF"])
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(7,26,61,0.6)", font=dict(color="#E8F0FF"))
        result["distribution"] = fig_to_json(fig)
        result["type"] = "regression"
    else:
        counts = df[target].value_counts()
        fig = px.pie(values=counts.values, names=counts.index.astype(str),
                     title=f"Class Distribution — {target}",
                     color_discrete_sequence=["#1677FF", "#19C6FF", "#7257FF", "#041127", "#F5F8FF"])
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="#E8F0FF"))
        result["distribution"] = fig_to_json(fig)
        result["type"] = "classification"
        result["imbalance"] = {
            "counts": counts.to_dict(),
            "pct": (counts / counts.sum() * 100).round(2).to_dict(),
        }
    return result
