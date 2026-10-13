from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict

PictureStyle = Literal["photo", "illustration"]
PictureStatus = Literal["queued", "planning", "rendering", "checking", "ready", "failed"]


class PictureRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    style: PictureStyle = "photo"


class PictureJob(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    lesson_id: UUID
    status: PictureStatus
    style: PictureStyle
    model: str
    created_at: datetime
    updated_at: datetime
    asset_id: UUID | None = None
    error: str | None = None
