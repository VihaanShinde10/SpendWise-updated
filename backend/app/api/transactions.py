from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, UploadFile, File, Query
from app.dependencies import get_current_user
from app.db.client import get_supabase
from app.utils.csv_parser import parse_bank_statement
from app.background.categorise import process_transactions_batch
from app.ml.layers.layer0_prep import prepare_transaction
from datetime import datetime
from typing import Optional
import uuid

router = APIRouter()


@router.post("/upload", status_code=202)
async def upload_transactions(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user)
):
    """Upload a CSV or Excel bank statement. Returns 202 immediately; ML runs in background."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    ext = file.filename.lower().split('.')[-1]
    if ext not in ('csv', 'xlsx', 'xls'):
        raise HTTPException(status_code=400, detail="Only CSV and Excel files are supported")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    # Parse the file
    try:
        raw_transactions = parse_bank_statement(content, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if not raw_transactions:
        raise HTTPException(status_code=422, detail="No valid transactions found in the file")

    supabase = get_supabase()

    # Ensure user profile exists
    try:
        supabase.table("profiles").upsert({
            "id": user_id,
            "email": f"user_{user_id[:8]}@spendwise.app",  # fallback email
        }, on_conflict="id").execute()
    except Exception:
        pass

    # Save raw transactions with 'pending' status
    saved_ids = []
    for txn in raw_transactions:
        # Run Layer 0 preprocessing
        try:
            prepped = prepare_transaction(
                raw_description=txn['raw_description'],
                amount=txn['amount'],
                direction=txn['direction'],
                payment_method=txn.get('payment_method', 'OTHER'),
                transaction_date=txn['transaction_date'],
                balance=txn.get('balance'),
            )
            cleaned_description = prepped.cleaned_description
            merchant_name = prepped.merchant_name
            is_low = prepped.is_low_descriptiveness
        except Exception:
            cleaned_description = txn['raw_description']
            merchant_name = txn['raw_description'][:30]
            is_low = False

        txn_record = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "raw_description": txn['raw_description'],
            "amount": txn['amount'],
            "direction": txn['direction'],
            "balance": txn.get('balance'),
            "transaction_date": txn['transaction_date'].isoformat(),
            "payment_method": txn.get('payment_method', 'OTHER'),
            "cleaned_description": cleaned_description,
            "merchant_name": merchant_name,
            "is_low_descriptiveness": is_low,
            "processing_status": "pending",
        }

        try:
            resp = supabase.table("transactions").insert(txn_record).execute()
            if resp.data:
                saved_ids.append(resp.data[0]['id'])
        except Exception as e:
            # Skip duplicates or constraint violations
            pass

    if not saved_ids:
        raise HTTPException(status_code=422, detail="No new transactions could be saved (possible duplicates)")

    # Schedule background ML processing
    background_tasks.add_task(
        process_transactions_batch,
        user_id=user_id,
        transaction_ids=saved_ids
    )

    return {
        "message": f"Upload received. Categorising {len(saved_ids)} transactions in background.",
        "transaction_count": len(saved_ids),
        "status": "processing"
    }


@router.get("")
async def list_transactions(
    status: Optional[str] = Query(None),
    category_id: Optional[str] = Query(None),
    direction: Optional[str] = Query(None),
    needs_review: Optional[bool] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    user_id: str = Depends(get_current_user)
):
    """List transactions for the authenticated user with optional filters."""
    supabase = get_supabase()
    offset = (page - 1) * page_size

    query = supabase.table("transactions").select("*").eq("user_id", user_id)

    if status:
        query = query.eq("processing_status", status)
    if category_id:
        query = query.eq("category_id", category_id)
    if direction:
        query = query.eq("direction", direction)
    if needs_review is not None:
        query = query.eq("needs_review", needs_review)

    query = query.order("transaction_date", desc=True).range(offset, offset + page_size - 1)
    resp = query.execute()

    return {"data": resp.data or [], "page": page, "page_size": page_size}


@router.get("/review")
async def get_review_queue(
    user_id: str = Depends(get_current_user)
):
    """Get transactions that need manual category review."""
    supabase = get_supabase()
    resp = supabase.table("transactions").select("*").eq("user_id", user_id).eq(
        "needs_review", True
    ).eq("user_corrected", False).order("created_at", desc=True).limit(100).execute()
    return {"data": resp.data or []}


@router.get("/{txn_id}")
async def get_transaction(
    txn_id: str,
    user_id: str = Depends(get_current_user)
):
    supabase = get_supabase()
    resp = supabase.table("transactions").select("*").eq("id", txn_id).eq("user_id", user_id).single().execute()
    if not resp.data:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return resp.data


@router.patch("/{txn_id}/category")
async def update_transaction_category(
    txn_id: str,
    body: dict,
    user_id: str = Depends(get_current_user)
):
    """Manual category correction. Marks transaction as user-corrected."""
    category_id = body.get("category_id")
    if not category_id:
        raise HTTPException(status_code=400, detail="category_id is required")

    supabase = get_supabase()
    resp = supabase.table("transactions").update({
        "user_category_id": category_id,
        "category_id": category_id,
        "user_corrected": True,
        "needs_review": False,
        "category_source": "manual",
    }).eq("id", txn_id).eq("user_id", user_id).execute()

    if not resp.data:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return resp.data[0]
