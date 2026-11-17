"""
Automated PDF Report Generation
"""
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse
from pathlib import Path
from datetime import datetime
import json
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, white, black
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT

from app.routes.auth import get_current_user
from app.database import get_db
from app.routes.cleaning import load_project_df
from app.services.profiler import profile_dataset

router = APIRouter()
REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

NAVY = HexColor("#041127")
BLUE = HexColor("#1677FF")
CYAN = HexColor("#19C6FF")
PURPLE = HexColor("#7257FF")
LIGHT = HexColor("#F5F8FF")


@router.post("/generate/{project_id}")
async def generate_report(project_id: int, current_user: dict = Depends(get_current_user)):
    with get_db() as conn:
        c = conn.cursor()
        c.execute(
            """SELECT p.*, d.filename, d.rows, d.columns, d.health_score
               FROM projects p LEFT JOIN datasets d ON d.project_id = p.id
               WHERE p.id = ? AND p.user_id = ?""",
            (project_id, current_user["user_id"])
        )
        proj = c.fetchone()
        if not proj:
            raise HTTPException(404, "Project not found")

        c.execute(
            "SELECT algorithm, metrics, timestamp FROM experiments WHERE project_id = ?",
            (project_id,)
        )
        experiments = c.fetchall()

    # Profile
    try:
        df, _, _ = load_project_df(project_id, current_user["user_id"])
        profile = profile_dataset(df, proj["filename"] or "dataset")
    except Exception:
        profile = {"rows": proj["rows"], "columns": proj["columns"], "health_score": proj["health_score"]}

    filename = f"DataPilot_Report_{project_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
    filepath = REPORTS_DIR / filename

    doc = SimpleDocTemplate(str(filepath), pagesize=A4,
                            rightMargin=0.6*inch, leftMargin=0.6*inch,
                            topMargin=0.6*inch, bottomMargin=0.6*inch)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleNavy", parent=styles["Title"],
                              textColor=NAVY, fontSize=22, spaceAfter=12, alignment=TA_CENTER))
    styles.add(ParagraphStyle(name="H2Blue", parent=styles["Heading2"],
                              textColor=BLUE, fontSize=14, spaceBefore=14, spaceAfter=8))
    styles.add(ParagraphStyle(name="Body", parent=styles["Normal"],
                              textColor=black, fontSize=10, leading=14))
    styles.add(ParagraphStyle(name="Small", parent=styles["Normal"],
                              textColor=HexColor("#555555"), fontSize=8))

    story = []
    story.append(Paragraph("DataPilot AI", styles["TitleNavy"]))
    story.append(Paragraph("Project Report — Upload. Analyze. Train. Predict. Understand.", styles["Body"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"<b>Project:</b> {proj['name']} &nbsp;&nbsp; <b>Domain:</b> {proj['domain'] or 'General'}", styles["Body"]))
    story.append(Paragraph(f"<b>Generated:</b> {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} &nbsp;&nbsp; <b>User:</b> {current_user['username']}", styles["Small"]))
    story.append(Spacer(1, 12))

    # Dataset Summary
    story.append(Paragraph("1. Dataset Summary", styles["H2Blue"]))
    data = [
        ["Metric", "Value"],
        ["Filename", str(proj["filename"] or "—")],
        ["Rows", str(profile.get("rows", "—"))],
        ["Columns", str(profile.get("columns", "—"))],
        ["Health Score", f"{profile.get('health_score', '—')}%"],
        ["Numerical Features", str(len(profile.get("numerical_features", [])))],
        ["Categorical Features", str(len(profile.get("categorical_features", [])))],
        ["Duplicates", str(profile.get("duplicates", "—"))],
    ]
    t = Table(data, colWidths=[2.5*inch, 3.5*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 1), (-1, -1), LIGHT),
        ("GRID", (0, 0), (-1, -1), 0.4, BLUE),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 12))

    # Health factors
    if profile.get("health_factors"):
        story.append(Paragraph("Data Quality Factors", styles["H2Blue"]))
        factors = profile["health_factors"]
        fdata = [["Factor", "Score"]] + [[k.replace("_", " ").title(), f"{v}%"] for k, v in factors.items()]
        ft = Table(fdata, colWidths=[2.5*inch, 1.5*inch])
        ft.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), PURPLE),
            ("TEXTCOLOR", (0, 0), (-1, 0), white),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.3, CYAN),
            ("PADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(ft)
        story.append(Spacer(1, 12))

    # Models
    story.append(Paragraph("2. Models & Evaluation", styles["H2Blue"]))
    if not experiments:
        story.append(Paragraph("No models trained yet.", styles["Body"]))
    else:
        for exp in experiments:
            metrics = json.loads(exp["metrics"] or "{}")
            story.append(Paragraph(f"<b>{exp['algorithm']}</b> — {exp['timestamp'][:19]}", styles["Body"]))
            mrows = [["Metric", "Value"]]
            for k, v in metrics.items():
                if k not in ("confusion_matrix", "report", "actual", "predicted", "projection"):
                    mrows.append([k.upper(), str(round(v, 4) if isinstance(v, float) else v)])
            mt = Table(mrows, colWidths=[2*inch, 2*inch])
            mt.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), BLUE),
                ("TEXTCOLOR", (0, 0), (-1, 0), white),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.3, HexColor("#CCCCCC")),
                ("PADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(mt)
            story.append(Spacer(1, 8))

    story.append(Spacer(1, 16))
    story.append(Paragraph("Generated by DataPilot AI — Upload. Analyze. Train. Predict. Understand.", styles["Small"]))
    story.append(Paragraph("This report is for informational purposes. Model performance depends on data quality and problem formulation.", styles["Small"]))

    doc.build(story)

    return {
        "success": True,
        "filename": filename,
        "path": f"/api/reports/download/{filename}",
        "message": "Report generated successfully",
    }


@router.get("/download/{filename}")
async def download_report(filename: str, current_user: dict = Depends(get_current_user)):
    # Basic safety
    if ".." in filename or "/" in filename:
        raise HTTPException(400, "Invalid filename")
    path = REPORTS_DIR / filename
    if not path.exists():
        raise HTTPException(404, "Report not found")
    return FileResponse(path, media_type="application/pdf", filename=filename)
