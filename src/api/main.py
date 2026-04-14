from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes.companies import router as companies_router
from .routes.ingest import router as ingest_router
from .routes.pipeline import router as pipeline_router
from .routes.portfolio import router as portfolio_router
from .routes.reports import router as reports_router
from .routes.signals import router as signals_router
from .routes.thesis import router as thesis_router


app = FastAPI(title="PERIS API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(companies_router, prefix="/companies", tags=["companies"])
app.include_router(pipeline_router, prefix="/pipeline", tags=["pipeline"])
app.include_router(portfolio_router, prefix="/portfolio", tags=["portfolio"])
app.include_router(signals_router, prefix="/signals", tags=["signals"])
app.include_router(reports_router, prefix="/reports", tags=["reports"])
app.include_router(thesis_router, prefix="/thesis", tags=["thesis"])
app.include_router(ingest_router, prefix="/ingest", tags=["ingest"])


@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "ok", "service": "peris-api"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
