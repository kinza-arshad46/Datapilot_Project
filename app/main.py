"""
DataPilot AI - Complete End-to-End Machine Learning Platform
Main FastAPI application entry point
"""
import os
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.routes import auth, upload, cleaning, eda, models, prediction, reports, dashboard

# Ensure directories exist
BASE_DIR = Path(__file__).resolve().parent.parent
for d in ["data", "models", "reports", "data/uploads", "data/projects"]:
    (BASE_DIR / d).mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="DataPilot AI",
    description="Upload. Analyze. Train. Predict. Understand. — Complete ML Platform",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static & Templates
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "app" / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(upload.router, prefix="/api/dataset", tags=["Dataset"])
app.include_router(cleaning.router, prefix="/api/cleaning", tags=["Data Cleaning"])
app.include_router(eda.router, prefix="/api/eda", tags=["EDA Studio"])
app.include_router(models.router, prefix="/api/models", tags=["Model Lab"])
app.include_router(prediction.router, prefix="/api/prediction", tags=["Prediction"])
app.include_router(reports.router, prefix="/api/reports", tags=["Reports"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["Dashboard"])


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/{page}", response_class=HTMLResponse)
async def pages(request: Request, page: str):
    allowed = [
        "home", "dataset", "cleaning", "eda", "model-lab",
        "comparison", "prediction", "explainability", "dashboard",
        "reports", "settings"
    ]
    if page in allowed:
        return templates.TemplateResponse("index.html", {"request": request, "page": page})
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    return {"status": "ok", "service": "DataPilot AI", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
