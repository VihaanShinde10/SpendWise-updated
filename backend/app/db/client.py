from supabase import create_client, Client
from app.config import get_settings
from functools import lru_cache

settings = get_settings()


@lru_cache()
def get_supabase() -> Client:
    """Return a singleton Supabase client using service role key (backend use only)."""
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)


def get_supabase_anon() -> Client:
    """Return a Supabase client using anon key (for user-scoped operations with RLS)."""
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_ANON_KEY)
