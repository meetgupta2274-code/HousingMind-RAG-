"""
FastAPI backend for the Housing Price RAG application.
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ingest import ingest_data, is_already_ingested, get_collection_count
from rag_engine import query_with_llm


# Track ingestion status
app_state = {"ingestion_status": "unknown", "doc_count": 0}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Check embedding status on startup."""
    if is_already_ingested():
        app_state["ingestion_status"] = "ready"
        app_state["doc_count"] = get_collection_count()
        print(f"✅ ChromaDB ready with {app_state['doc_count']} documents")
    else:
        app_state["ingestion_status"] = "not_ingested"
        print("⚠️ No embeddings found. Call POST /api/ingest to create them.")
    yield


app = FastAPI(
    title="India Housing Price RAG API",
    description="RAG-powered API for querying Indian real estate data",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS - allow frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Request / Response Models ----------

class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list
    error: bool = False


class IngestResponse(BaseModel):
    status: str
    message: str
    count: int


class HealthResponse(BaseModel):
    status: str
    ingestion_status: str
    doc_count: int


# ---------- Endpoints ----------

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with embedding status."""
    # Refresh status
    if is_already_ingested():
        app_state["ingestion_status"] = "ready"
        app_state["doc_count"] = get_collection_count()

    return HealthResponse(
        status="healthy",
        ingestion_status=app_state["ingestion_status"],
        doc_count=app_state["doc_count"],
    )


@app.head("/health")
@app.get("/health")
async def uptime_health():
    """Lightweight health endpoint for uptime monitors (HEAD/GET)."""
    return {"status": "ok"}


@app.post("/api/ingest", response_model=IngestResponse)
async def run_ingestion():
    """Trigger data ingestion into ChromaDB."""
    try:
        result = ingest_data()
        app_state["ingestion_status"] = "ready"
        app_state["doc_count"] = result["count"]
        return IngestResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/api/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Query the RAG pipeline with a natural language question."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if not is_already_ingested():
        return QueryResponse(
            answer="⚠️ Data has not been ingested yet. Please click 'Initialize Database' first.",
            sources=[],
            error=True,
        )

    try:
        result = query_with_llm(request.question.strip())
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
