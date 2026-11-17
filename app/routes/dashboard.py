"""
Dashboard KPIs and overview
"""
from fastapi import APIRouter, HTTPException, Depends
import json
from app.routes.auth import get_current_user
from app.database import get_db
from app.routes.cleaning import load_project_df
from app.services.profiler import profile_dataset

router = APIRouter()


@router.get("/{project_id}")
async def dashboard_data(project_id: int, current_user: dict = Depends(get_current_user)):
    with get_db() as conn:
        c = conn.cursor()
        c.execute(
            """SELECT p.*, d.rows, d.columns, d.health_score, d.filename
               FROM projects p
               LEFT JOIN datasets d ON d.project_id = p.id
               WHERE p.id = ? AND p.user_id = ?""",
            (project_id, current_user["user_id"])
        )
        proj = c.fetchone()
        if not proj:
            raise HTTPException(404, "Project not found")

        c.execute(
            "SELECT id, algorithm, metrics, timestamp FROM experiments WHERE project_id = ? ORDER BY timestamp DESC",
            (project_id,)
        )
        experiments = c.fetchall()

        c.execute(
            "SELECT COUNT(*) as cnt FROM predictions WHERE project_id = ?",
            (project_id,)
        )
        pred_count = c.fetchone()["cnt"]

        c.execute(
            "SELECT output, timestamp FROM predictions WHERE project_id = ? ORDER BY timestamp DESC LIMIT 5",
            (project_id,)
        )
        recent_preds = c.fetchall()

    # Best metric
    best = None
    best_score = -1
    model_perf = []
    for exp in experiments:
        metrics = json.loads(exp["metrics"] or "{}")
        score = metrics.get("accuracy") or metrics.get("r2") or metrics.get("silhouette") or 0
        model_perf.append({
            "algorithm": exp["algorithm"],
            "score": round(float(score), 4) if score else 0,
            "metrics": metrics,
        })
        if score and score > best_score:
            best_score = score
            best = exp["algorithm"]

    # Try load class distribution
    class_dist = {}
    try:
        df, _, _ = load_project_df(project_id, current_user["user_id"])
        profile = profile_dataset(df)
        class_dist = profile.get("class_distribution", {})
    except Exception:
        profile = {}

    return {
        "project": {
            "id": project_id,
            "name": proj["name"],
            "domain": proj["domain"],
            "problem_type": proj["problem_type"],
        },
        "kpis": {
            "rows": proj["rows"] or 0,
            "features": proj["columns"] or 0,
            "missing_pct": 0,  # simplified
            "model_count": len(experiments),
            "best_metric": round(best_score, 4) if best_score >= 0 else None,
            "best_model": best,
            "prediction_count": pred_count,
            "health_score": proj["health_score"] or 0,
        },
        "model_performance": model_perf,
        "class_distribution": class_dist,
        "recent_predictions": [
            {"output": json.loads(r["output"] or "{}"), "timestamp": r["timestamp"]}
            for r in recent_preds
        ],
        "filename": proj["filename"],
    }
