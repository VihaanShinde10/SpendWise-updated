from fastapi import APIRouter, Depends, Query
from app.dependencies import get_current_user
from app.db.client import get_supabase
from datetime import datetime, timedelta
from typing import Optional
from collections import defaultdict

router = APIRouter()


@router.get("/summary")
async def spending_summary(
    months: int = Query(1, ge=1, le=12),
    user_id: str = Depends(get_current_user)
):
    """Monthly spending summary — total in, total out, net."""
    supabase = get_supabase()
    since = (datetime.utcnow() - timedelta(days=30 * months)).isoformat()

    resp = supabase.table("transactions").select(
        "amount, direction, transaction_date, category_id"
    ).eq("user_id", user_id).in_("processing_status", ["completed", "processing"]).gte(
        "transaction_date", since
    ).execute()

    total_debit = sum(t['amount'] for t in (resp.data or []) if t['direction'] == 'debit')
    total_credit = sum(t['amount'] for t in (resp.data or []) if t['direction'] == 'credit')

    return {
        "total_spent": round(total_debit, 2),
        "total_received": round(total_credit, 2),
        "net": round(total_credit - total_debit, 2),
        "transaction_count": len(resp.data or []),
        "period_months": months,
    }


@router.get("/by-category")
async def spending_by_category(
    months: int = Query(1, ge=1, le=12),
    user_id: str = Depends(get_current_user)
):
    """Spending breakdown by category for debits only."""
    supabase = get_supabase()
    since = (datetime.utcnow() - timedelta(days=30 * months)).isoformat()

    txns_resp = supabase.table("transactions").select(
        "amount, direction, category_id"
    ).eq("user_id", user_id).eq("direction", "debit").in_(
        "processing_status", ["completed", "processing"]
    ).gte("transaction_date", since).execute()

    cats_resp = supabase.table("categories").select("id, name, icon, color").execute()
    cat_map = {c['id']: c for c in (cats_resp.data or [])}

    totals: dict = defaultdict(float)
    counts: dict = defaultdict(int)
    for t in (txns_resp.data or []):
        cid = t.get('category_id')
        if cid:
            totals[cid] += t['amount']
            counts[cid] += 1

    result = []
    grand_total = sum(totals.values())
    for cid, total in sorted(totals.items(), key=lambda x: -x[1]):
        cat = cat_map.get(cid, {})
        result.append({
            "category_id": cid,
            "category_name": cat.get('name', 'Unknown'),
            "icon": cat.get('icon', '📦'),
            "color": cat.get('color', '#D5DBDB'),
            "total": round(total, 2),
            "count": counts[cid],
            "percentage": round(total / grand_total * 100, 1) if grand_total > 0 else 0,
        })
    return result


@router.get("/trends")
async def spending_trends(
    months: int = Query(6, ge=2, le=24),
    user_id: str = Depends(get_current_user)
):
    """Month-over-month spending trends."""
    supabase = get_supabase()
    since = (datetime.utcnow() - timedelta(days=30 * months)).isoformat()

    resp = supabase.table("transactions").select(
        "amount, direction, transaction_date"
    ).eq("user_id", user_id).in_("processing_status", ["completed", "processing"]).gte(
        "transaction_date", since
    ).execute()

    monthly: dict = defaultdict(lambda: {"spent": 0.0, "received": 0.0, "count": 0})
    for t in (resp.data or []):
        try:
            dt = datetime.fromisoformat(t['transaction_date'].replace('Z', '+00:00'))
            key = dt.strftime("%Y-%m")
            if t['direction'] == 'debit':
                monthly[key]['spent'] += t['amount']
            else:
                monthly[key]['received'] += t['amount']
            monthly[key]['count'] += 1
        except Exception:
            pass

    result = []
    for key in sorted(monthly.keys()):
        m = monthly[key]
        result.append({
            "month": key,
            "spent": round(m['spent'], 2),
            "received": round(m['received'], 2),
            "count": m['count'],
        })
    return result


@router.get("/recurring")
async def recurring_payments(user_id: str = Depends(get_current_user)):
    """Detected recurring payments."""
    supabase = get_supabase()
    resp = supabase.table("transactions").select(
        "merchant_name, amount, transaction_date, category_id, recurrence_strength, is_recurring"
    ).eq("user_id", user_id).eq("is_recurring", True).order(
        "recurrence_strength", desc=True
    ).limit(50).execute()
    return resp.data or []


@router.get("/cold-start-status")
async def cold_start_status(user_id: str = Depends(get_current_user)):
    """Return user's data stage: cold (<15), developing (15-50), established (>50)."""
    supabase = get_supabase()
    resp = supabase.table("transactions").select("id").eq("user_id", user_id).eq(
        "processing_status", "completed"
    ).execute()
    count = len(resp.data or [])

    if count < 15:
        stage = "cold"
        coverage_pct = 74
    elif count < 50:
        stage = "developing"
        coverage_pct = 83
    else:
        stage = "established"
        coverage_pct = 91

    return {
        "transaction_count": count,
        "stage": stage,
        "expected_coverage_pct": coverage_pct,
        "next_milestone": max(0, 15 - count) if stage == "cold" else max(0, 50 - count),
    }
