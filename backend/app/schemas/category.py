from pydantic import BaseModel
from typing import Optional
from uuid import UUID
from datetime import datetime


class CategoryOut(BaseModel):
    id: UUID
    name: str
    icon: Optional[str]
    color: Optional[str]
    is_system: Optional[bool]
    parent_id: Optional[UUID]
    created_at: datetime

    class Config:
        from_attributes = True


class CategoryCreate(BaseModel):
    name: str
    icon: Optional[str] = "📦"
    color: Optional[str] = "#D5DBDB"
    parent_id: Optional[UUID] = None
