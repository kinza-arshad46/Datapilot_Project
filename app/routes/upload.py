"""
Dataset Upload & Profiling
"""
import os
import uuid
from pathlib import Path
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form, Request
from fastapi.responses import JSONResponse
from jose import JWTError, jwt
import pandas as pd
import json
import os

from app.routes.auth import get_current_user, SECRET_KEY, ALGORITHM
from app.services.profiler import profile_dataset
from app.database import get_db

router = APIRouter()

UPLOAD_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
MAX_SIZE_MB = 50


def _user_from_token(token: str):
    """Decode JWT and return user dict or None."""
    if not token:
        return None
    try:
        # Allow "Bearer xxx" or raw token
        if token.lower().startswith("bearer "):
            token = token[7:].strip()
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        sub = payload.get("sub")
        if sub is None:
            return None
        return {"user_id": int(sub), "username": payload.get("username")}
    except Exception:
        return None


@router.post("/upload")
async def upload_dataset(
    request: Request,
    file: UploadFile = File(...),
    project_name: str = Form("My Project"),
    domain: str = Form("general"),
    token: str = Form(None),
):
    # Auth: prefer Authorization header, fallback to form field "token"
    current_user = None
    auth_header = request.headers.get("authorization") or request.headers.get("Authorization")
    if auth_header:
        current_user = _user_from_token(auth_header)
    if current_user is None and token:
        current_user = _user_from_token(token)
    if current_user is None:
        raise HTTPException(status_code=401, detail="Not authenticated. Please Sign In again.")
    # Validate extension
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")

    # Read content
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_SIZE_MB:
        raise HTTPException(status_code=400, detail=f"File too large. Max {MAX_SIZE_MB}MB")

    # Save safely
    safe_name = f"{uuid.uuid4().hex}{ext}"
    filepath = UPLOAD_DIR / safe_name
    with open(filepath, "wb") as f:
        f.write(content)

    # Load with pandas
    try:
        if ext == ".csv":
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
    except Exception as e:
        filepath.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Cannot read file: {str(e)}")

    if df.empty or df.shape[0] < 5:
        filepath.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Dataset too small or empty (min 5 rows)")

    # Profile
    profile = profile_dataset(df, filename=file.filename or safe_name)

    # Create project + dataset records
    with get_db() as conn:
        c = conn.cursor()
        c.execute(
            "INSERT INTO projects (user_id, name, domain, created_at) VALUES (?, ?, ?, ?)",
            (current_user["user_id"], project_name, domain, datetime.utcnow().isoformat())
        )
        project_id = c.lastrowid
        c.execute(
            """INSERT INTO datasets (project_id, filename, filepath, rows, columns, schema_json, health_score, upload_meta, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                project_id,
                file.filename or safe_name,
                str(filepath),
                profile["rows"],
                profile["columns"],
                json.dumps(profile["dtypes"]),
                profile["health_score"],
                json.dumps({"size_mb": round(size_mb, 2)}),
                datetime.utcnow().isoformat(),
            )
        )
        dataset_id = c.lastrowid
        conn.commit()

    # Cache df path for session (simple: store path in response, client keeps project_id)
    return {
        "success": True,
        "project_id": project_id,
        "dataset_id": dataset_id,
        "profile": profile,
        "message": "Dataset uploaded and profiled successfully",
    }


@router.get("/profile/{project_id}")
async def get_profile(project_id: int, current_user: dict = Depends(get_current_user)):
    with get_db() as conn:
        c = conn.cursor()
        c.execute(
            """SELECT d.*, p.user_id, p.name as project_name, p.domain
               FROM datasets d JOIN projects p ON d.project_id = p.id
               WHERE d.project_id = ? AND p.user_id = ?""",
            (project_id, current_user["user_id"])
        )
        row = c.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Project/dataset not found")

    # Re-profile from file for fresh data
    try:
        path = row["filepath"]
        if path.endswith(".csv"):
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path)
        profile = profile_dataset(df, filename=row["filename"])
        profile["project_id"] = project_id
        profile["project_name"] = row["project_name"]
        profile["domain"] = row["domain"]
        return profile
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load dataset: {str(e)}")


@router.get("/preview/{project_id}")
async def preview_data(project_id: int, n: int = 10, current_user: dict = Depends(get_current_user)):
    with get_db() as conn:
        c = conn.cursor()
        c.execute(
            """SELECT d.filepath FROM datasets d JOIN projects p ON d.project_id = p.id
               WHERE d.project_id = ? AND p.user_id = ?""",
            (project_id, current_user["user_id"])
        )
        row = c.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Not found")

    path = row["filepath"]
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    return {
        "columns": df.columns.tolist(),
        "data": df.head(n).fillna("").astype(str).to_dict(orient="records"),
        "total_rows": len(df),
    }
