from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from .database import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)

    # Status lifecycle: pending → running → done | failed | cancelled
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending", index=True)

    workflow: Mapped[str] = mapped_column(String(64), nullable=False)
    prompt: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Arbitrary sampler/workflow params (steps, denoise, seed, …)
    params: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    input_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_path: Mapped[str | None] = mapped_column(Text, nullable=True)

    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_sec: Mapped[float | None] = mapped_column(Float, nullable=True)

    def __repr__(self) -> str:
        return f"<Job id={self.id[:8]} status={self.status} workflow={self.workflow}>"

    def mark_running(self) -> None:
        self.status = "running"
        self.started_at = _now()

    def mark_done(self, output_path: str) -> None:
        self.status = "done"
        self.output_path = output_path
        self.finished_at = _now()
        if self.started_at:
            self.duration_sec = (self.finished_at - self.started_at).total_seconds()

    def mark_failed(self, error: str) -> None:
        self.status = "failed"
        self.error = error
        self.finished_at = _now()
        if self.started_at:
            self.duration_sec = (self.finished_at - self.started_at).total_seconds()

    def mark_cancelled(self) -> None:
        self.status = "cancelled"
        self.finished_at = _now()
