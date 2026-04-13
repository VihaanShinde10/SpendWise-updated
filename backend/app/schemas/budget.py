from pydantic import BaseModel
from typing import Optional, List
from uuid import UUID
from datetime import date


class BudgetCreate(BaseModel):
    category_id: UUID
    amount: float
    period: str = "monthly"
    start_date: date
    end_date: Optional[date] = None


class BudgetOut(BaseModel):
    id: UUID
    user_id: UUID
    category_id: UUID
    amount: float
    period: str
    start_date: date
    end_date: Optional[date]
    created_at: str

    class Config:
        from_attributes = True


class BudgetStatus(BaseModel):
    budget_id: UUID
    category_id: UUID
    category_name: str
    budgeted: float
    spent: float
    remaining: float
    percentage_used: float
    period: str
