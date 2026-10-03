"""
Tests for JWT auth routes — register, login, me.
"""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from uncomfy.database import Base, get_session
from uncomfy.models import User

TEST_DB = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="module")
async def db_engine():
    engine = create_async_engine(TEST_DB)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def app(db_engine):
    from uncomfy import main as m
    Session = async_sessionmaker(db_engine, expire_on_commit=False)

    async def _override():
        async with Session() as s:
            yield s

    m.app.dependency_overrides[get_session] = _override
    with (
        patch.object(m.broker, "startup", new=AsyncMock()),
        patch.object(m.broker, "shutdown", new=AsyncMock()),
    ):
        yield m.app
    m.app.dependency_overrides.clear()


@pytest.fixture
async def client(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------

class TestRegister:
    async def test_register_success(self, client):
        r = await client.post("/auth/register", data={"username": "alice", "password": "secret123"})
        assert r.status_code == 201
        assert r.json()["username"] == "alice"

    async def test_duplicate_username(self, client):
        await client.post("/auth/register", data={"username": "bob", "password": "secret123"})
        r = await client.post("/auth/register", data={"username": "bob", "password": "other123"})
        assert r.status_code == 409

    async def test_short_username(self, client):
        r = await client.post("/auth/register", data={"username": "ab", "password": "secret123"})
        assert r.status_code == 422

    async def test_short_password(self, client):
        r = await client.post("/auth/register", data={"username": "carol", "password": "12345"})
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# Token (OAuth2)
# ---------------------------------------------------------------------------

class TestToken:
    async def test_login_returns_token(self, client):
        await client.post("/auth/register", data={"username": "dave", "password": "hunter2!"})
        r = await client.post("/auth/token", data={"username": "dave", "password": "hunter2!"})
        assert r.status_code == 200
        data = r.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    async def test_wrong_password(self, client):
        await client.post("/auth/register", data={"username": "eve", "password": "correct99"})
        r = await client.post("/auth/token", data={"username": "eve", "password": "wrong"})
        assert r.status_code == 401

    async def test_unknown_user(self, client):
        r = await client.post("/auth/token", data={"username": "nobody", "password": "x"})
        assert r.status_code == 401


# ---------------------------------------------------------------------------
# /auth/me
# ---------------------------------------------------------------------------

class TestMe:
    async def _get_token(self, client, username="frank", password="pass1234") -> str:
        await client.post("/auth/register", data={"username": username, "password": password})
        r = await client.post("/auth/token", data={"username": username, "password": password})
        return r.json()["access_token"]

    async def test_me_with_valid_token(self, client):
        token = await self._get_token(client, "grace", "pass1234")
        r = await client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        assert r.json()["username"] == "grace"

    async def test_me_without_token(self, client):
        r = await client.get("/auth/me")
        assert r.status_code == 401

    async def test_me_with_invalid_token(self, client):
        r = await client.get("/auth/me", headers={"Authorization": "Bearer bad.token.here"})
        assert r.status_code == 401


# ---------------------------------------------------------------------------
# auth helpers
# ---------------------------------------------------------------------------

class TestAuthHelpers:
    def test_hash_and_verify(self):
        from uncomfy.auth import hash_password, verify_password
        h = hash_password("my_secret")
        assert verify_password("my_secret", h)
        assert not verify_password("wrong", h)

    def test_create_and_decode_token(self):
        from uncomfy.auth import create_access_token, _decode_token
        token = create_access_token("user-123", "heidi")
        payload = _decode_token(token)
        assert payload["sub"] == "user-123"
        assert payload["name"] == "heidi"

    def test_bad_token_returns_empty(self):
        from uncomfy.auth import _decode_token
        assert _decode_token("bad.token") == {}
