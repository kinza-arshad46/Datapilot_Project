"""
Authentication routes - Login / Register / Session
"""
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from jose import JWTError, jwt
from typing import Optional
import os

from app.database import get_db, init_db, dict_from_row

router = APIRouter()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer(auto_error=False)

SECRET_KEY = os.getenv("SECRET_KEY", "datapilot-ai-super-secret-key-change-in-production-2024")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# Initialize DB on import
init_db()


class UserCreate(BaseModel):
    username: str
    email: Optional[str] = None
    password: str


class UserLogin(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    user_id: int


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    # JWT 'sub' should be a string
    if "sub" in to_encode:
        to_encode["sub"] = str(to_encode["sub"])
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        sub = payload.get("sub")
        username = payload.get("username")
        if sub is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        try:
            user_id = int(sub)
        except (TypeError, ValueError):
            raise HTTPException(status_code=401, detail="Invalid token subject")
        return {"user_id": user_id, "username": username}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


@router.post("/register", response_model=Token)
async def register(user: UserCreate):
    if len(user.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    if len(user.username) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters")

    with get_db() as conn:
        c = conn.cursor()
        # Check existing
        c.execute("SELECT id FROM users WHERE username = ?", (user.username,))
        if c.fetchone():
            raise HTTPException(status_code=400, detail="Username already registered")
        if user.email:
            c.execute("SELECT id FROM users WHERE email = ?", (user.email,))
            if c.fetchone():
                raise HTTPException(status_code=400, detail="Email already registered")

        hash_ = get_password_hash(user.password)
        c.execute(
            "INSERT INTO users (username, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (user.username, user.email, hash_, datetime.utcnow().isoformat())
        )
        conn.commit()
        user_id = c.lastrowid

    token = create_access_token({"sub": user_id, "username": user.username})
    return Token(access_token=token, username=user.username, user_id=user_id)


@router.post("/login", response_model=Token)
async def login(user: UserLogin):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = ?", (user.username,))
        row = c.fetchone()
        if not row or not verify_password(user.password, row["password_hash"]):
            raise HTTPException(status_code=401, detail="Incorrect username or password")
        user_id = row["id"]
        username = row["username"]

    token = create_access_token({"sub": user_id, "username": username})
    return Token(access_token=token, username=username, user_id=user_id)


@router.get("/me")
async def me(current_user: dict = Depends(get_current_user)):
    return current_user


@router.post("/logout")
async def logout():
    # Client should discard token
    return {"message": "Logged out successfully"}
