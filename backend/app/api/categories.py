from fastapi import APIRouter, HTTPException, Depends
from app.dependencies import get_current_user
from app.db.client import get_supabase
from app.schemas.category import CategoryCreate

router = APIRouter()


@router.get("")
async def list_categories(user_id: str = Depends(get_current_user)):
    """List all categories: system-wide + user-specific."""
    supabase = get_supabase()
    resp = supabase.table("categories").select("*").or_(
        f"is_system.eq.true,user_id.eq.{user_id}"
    ).order("is_system", desc=True).execute()
    return resp.data or []


@router.post("", status_code=201)
async def create_category(
    body: CategoryCreate,
    user_id: str = Depends(get_current_user)
):
    supabase = get_supabase()
    resp = supabase.table("categories").insert({
        "user_id": user_id,
        "name": body.name,
        "icon": body.icon,
        "color": body.color,
        "parent_id": str(body.parent_id) if body.parent_id else None,
        "is_system": False,
    }).execute()
    return resp.data[0]


@router.delete("/{cat_id}")
async def delete_category(
    cat_id: str,
    user_id: str = Depends(get_current_user)
):
    supabase = get_supabase()
    # Only allow deleting user-owned categories
    resp = supabase.table("categories").delete().eq("id", cat_id).eq("user_id", user_id).eq("is_system", False).execute()
    return {"deleted": True}
