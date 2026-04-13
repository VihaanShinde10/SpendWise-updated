from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from app.config import get_settings
from app.api import auth, transactions, categories, analytics, budgets, ml

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: pre-load ML models into memory. Shutdown: clean up."""
    logger.info("SpendWise backend starting up...")
    logger.info("Pre-loading E5 embedding model (this takes ~30s on first run)...")
    try:
        from app.ml.layers.layer1_semantic import get_e5_model
        get_e5_model()
        logger.success("E5 model loaded successfully.")
    except Exception as e:
        logger.warning(f"E5 model not loaded at startup (will load on first request): {e}")

    logger.info("BART zero-shot model will be loaded lazily on first use.")
    logger.success("SpendWise API ready.")
    yield
    logger.info("SpendWise backend shutting down.")


app = FastAPI(
    title="SpendWise API",
    description="Hybrid UPI Transaction Categorisation System — 5-Layer ML Pipeline",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Register all routers
app.include_router(auth.router,         prefix="/api/auth",         tags=["Auth"])
app.include_router(transactions.router, prefix="/api/transactions", tags=["Transactions"])
app.include_router(categories.router,   prefix="/api/categories",   tags=["Categories"])
app.include_router(analytics.router,    prefix="/api/analytics",    tags=["Analytics"])
app.include_router(budgets.router,      prefix="/api/budgets",      tags=["Budgets"])
app.include_router(ml.router,           prefix="/api/ml",           tags=["ML"])


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok", "version": "1.0.0", "service": "SpendWise API"}
