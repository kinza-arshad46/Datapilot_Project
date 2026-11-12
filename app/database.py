"""
Simple SQLite database setup for users, projects, experiments
"""
import sqlite3
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime
import json

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "datapilot.db"


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_db() as conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                domain TEXT DEFAULT 'general',
                problem_type TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL,
                rows INTEGER,
                columns INTEGER,
                schema_json TEXT,
                health_score REAL,
                upload_meta TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects(id)
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                algorithm TEXT NOT NULL,
                parameters TEXT,
                metrics TEXT,
                model_path TEXT,
                preprocessing_path TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects(id)
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                experiment_id INTEGER,
                input_summary TEXT,
                output TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects(id)
            )
        """)
        # Default demo user (password: demo123)
        try:
            from passlib.context import CryptContext
            pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")
            hash_ = pwd.hash("demo123")
            c.execute(
                "INSERT OR IGNORE INTO users (username, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
                ("demo", "demo@datapilot.ai", hash_, datetime.utcnow().isoformat())
            )
        except Exception:
            pass
        conn.commit()


@contextmanager
def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def dict_from_row(row):
    return dict(row) if row else None
