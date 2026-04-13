from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from uuid import UUID


class TransactionCreate(BaseModel):
    raw_description: str
    amount: float
    direction: str = Field(..., pattern="^(debit|credit)$")
    payment_method: Optional[str] = "OTHER"
    transaction_date: datetime
    balance: Optional[float] = None


class TransactionOut(BaseModel):
    id: UUID
    user_id: UUID
    raw_description: str
    amount: float
    direction: str
    balance: Optional[float]
    transaction_date: datetime
    payment_method: Optional[str]
    cleaned_description: Optional[str]
    merchant_name: Optional[str]
    category_id: Optional[UUID]
    category_source: Optional[str]
    confidence_score: Optional[float]
    gating_alpha: Optional[float]
    is_recurring: Optional[bool]
    recurrence_strength: Optional[float]
    needs_review: Optional[bool]
    user_corrected: Optional[bool]
    processing_status: str
    processed_at: Optional[datetime]
    created_at: datetime

    class Config:
        from_attributes = True


class TransactionCategoryUpdate(BaseModel):
    category_id: UUID


class UploadResponse(BaseModel):
    message: str
    transaction_count: int
    status: str
    job_id: Optional[str] = None
