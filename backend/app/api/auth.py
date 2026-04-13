from fastapi import APIRouter, HTTPException
from app.db.client import get_supabase
from pydantic import BaseModel, EmailStr

router = APIRouter()


class RegisterRequest(BaseModel):
    email: str
    password: str
    display_name: str = ""


class LoginRequest(BaseModel):
    email: str
    password: str


@router.post("/register", status_code=201)
async def register(body: RegisterRequest):
    """
    Register a new user via Supabase Auth.
    Note: In production, the frontend should call Supabase directly.
    This endpoint exists for testing/docs purposes.
    """
    supabase = get_supabase()
    try:
        resp = supabase.auth.sign_up({
            "email": body.email,
            "password": body.password,
        })
        if resp.user:
            # Create profile
            try:
                supabase.table("profiles").upsert({
                    "id": str(resp.user.id),
                    "email": body.email,
                    "display_name": body.display_name,
                }, on_conflict="id").execute()
            except Exception:
                pass
            return {
                "user_id": str(resp.user.id),
                "email": resp.user.email,
                "message": "Registration successful. Check your email for confirmation."
            }
        raise HTTPException(status_code=400, detail="Registration failed")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/login")
async def login(body: LoginRequest):
    """
    Login with email/password via Supabase Auth.
    Returns: access_token, refresh_token, user_id
    """
    supabase = get_supabase()
    try:
        resp = supabase.auth.sign_in_with_password({
            "email": body.email,
            "password": body.password,
        })
        if resp.session:
            return {
                "access_token": resp.session.access_token,
                "refresh_token": resp.session.refresh_token,
                "token_type": "bearer",
                "user_id": str(resp.user.id),
                "email": resp.user.email,
            }
        raise HTTPException(status_code=401, detail="Invalid credentials")
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))


@router.post("/refresh")
async def refresh_token(body: dict):
    """Refresh an expired access token."""
    supabase = get_supabase()
    refresh_tk = body.get("refresh_token")
    if not refresh_tk:
        raise HTTPException(status_code=400, detail="refresh_token is required")
    try:
        resp = supabase.auth.refresh_session(refresh_tk)
        return {
            "access_token": resp.session.access_token,
            "refresh_token": resp.session.refresh_token,
        }
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))
