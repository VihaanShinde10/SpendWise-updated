from fastapi import APIRouter, Depends, HTTPException
from app.dependencies import get_current_user
from app.db.client import get_supabase
from app.schemas.budget import BudgetCreate
from datetime import datetime, timedelta

router = APIRouter()


@router.get("")
async def list_budgets(user_id: str = Depends(get_current_user)):
    supabase = get_supabase()
    resp = supabase.table("budgets").select("*").eq("user_id", user_id).execute()
    return resp.data or []


@router.post("", status_code=201)
async def create_budget(body: BudgetCreate, user_id: str = Depends(get_current_user)):
    supabase = get_supabase()
    resp = supabase.table("budgets").insert({
        "user_id": user_id,
        "category_id": str(body.category_id),
        "amount": body.amount,
        "period": body.period,
        "start_date": body.start_date.isoformat(),
        "end_date": body.end_date.isoformat() if body.end_date else None,
    }).execute()
    return resp.data[0]


@router.delete("/{budget_id}")
async def delete_budget(budget_id: str, user_id: str = Depends(get_current_user)):
    supabase = get_supabase()
    supabase.table("budgets").delete().eq("id", budget_id).eq("user_id", user_id).execute()
    return {"deleted": True}


@router.get("/status")
async def budget_status(user_id: str = Depends(get_current_user)):
    """Compare budget vs actual spending for current month."""
    supabase = get_supabase()
    budgets_resp = supabase.table("budgets").select("*").eq("user_id", user_id).execute()
    budgets = budgets_resp.data or []

    # Get current month spending by category
    month_start = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    txns_resp = supabase.table("transactions").select(
        "amount, direction, category_id"
    ).eq("user_id", user_id).eq("direction", "debit").eq(
        "processing_status", "completed"
    ).gte("transaction_date", month_start.isoformat()).execute()

    spent_by_cat: dict = {}
    for t in (txns_resp.data or []):
        cid = t.get('category_id')
        if cid:
            spent_by_cat[cid] = spent_by_cat.get(cid, 0) + t['amount']

    # Load category names
    cats_resp = supabase.table("categories").select("id, name").execute()
    cat_map = {c['id']: c['name'] for c in (cats_resp.data or [])}

    result = []
    for b in budgets:
        cid = b['category_id']
        spent = spent_by_cat.get(cid, 0.0)
        budgeted = b['amount']
        result.append({
            "budget_id": b['id'],
            "category_id": cid,
            "category_name": cat_map.get(cid, "Unknown"),
            "budgeted": round(budgeted, 2),
            "spent": round(spent, 2),
            "remaining": round(budgeted - spent, 2),
            "percentage_used": round(spent / budgeted * 100, 1) if budgeted > 0 else 0,
            "period": b['period'],
        })
    return result
