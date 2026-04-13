from fastapi import Depends, HTTPException, status
from typing import Optional
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from app.config import get_settings
from app.db.client import get_supabase

settings = get_settings()
bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> str:
    """
    Validate Supabase JWT using the official client and return the user_id.
    """
    # Debug: Print if headers exist
    print(f"DEBUG AUTH: Credentials found? {credentials is not None}")
    
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials missing in headers",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = credentials.credentials
    supabase = get_supabase()
    
    try:
        # Ask Supabase directly if the token is valid
        resp = supabase.auth.get_user(token)
        if not resp.user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session")
        return str(resp.user.id)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication failed: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )
