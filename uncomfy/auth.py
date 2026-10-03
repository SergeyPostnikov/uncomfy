from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import bcrypt as _bcrypt_lib
from jose import JWTError, jwt
from fastapi import Cookie, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .database import get_session
from .models import User

SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production-please")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

_oauth2 = OAuth2PasswordBearer(tokenUrl="/auth/token", auto_error=False)


# ---------------------------------------------------------------------------
# Passwords
# ---------------------------------------------------------------------------

def hash_password(plain: str) -> str:
    return _bcrypt_lib.hashpw(plain.encode(), _bcrypt_lib.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return _bcrypt_lib.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# JWT
# ---------------------------------------------------------------------------

def create_access_token(user_id: str, username: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode(
        {"sub": user_id, "name": username, "exp": expire},
        SECRET_KEY,
        algorithm=ALGORITHM,
    )


def _decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return {}


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------

async def _resolve_user(
    token: str | None,
    session: AsyncSession,
) -> User | None:
    if not token:
        return None
    payload = _decode_token(token)
    user_id = payload.get("sub")
    if not user_id:
        return None
    return await session.get(User, user_id)


async def get_current_user(
    bearer: str | None = Depends(_oauth2),
    access_token: str | None = Cookie(default=None),
    session: AsyncSession = Depends(get_session),
) -> User:
    """Reads JWT from Authorization header OR access_token cookie."""
    token = bearer or access_token
    user = await _resolve_user(token, session)
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


async def get_optional_user(
    bearer: str | None = Depends(_oauth2),
    access_token: str | None = Cookie(default=None),
    session: AsyncSession = Depends(get_session),
) -> User | None:
    """Like get_current_user but returns None instead of 401 (for HTML pages)."""
    token = bearer or access_token
    return await _resolve_user(token, session)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

async def get_user_by_username(session: AsyncSession, username: str) -> User | None:
    result = await session.execute(select(User).where(User.username == username))
    return result.scalar_one_or_none()
