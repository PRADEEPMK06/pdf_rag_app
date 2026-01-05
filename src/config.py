"""
Configuration and Data Models
"""
import os
from pathlib import Path
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# =============================================================================
# Paths Configuration
# =============================================================================

BASE_DIR = Path(__file__).parent  # src/
PROJECT_DIR = BASE_DIR.parent  # pdf_rag_app/
FRONTEND_DIR = BASE_DIR  # index.html is in src/
STORAGE_DIR = PROJECT_DIR / "storage"
DATA_DIR = STORAGE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
DOCUMENTS_DIR = DATA_DIR / "documents"

# Create all directories
for dir_path in [UPLOAD_DIR, PROCESSED_DIR, INDEX_DIR, DOCUMENTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Processing Configuration
# =============================================================================

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
MIN_CHUNK_SIZE = 50

# =============================================================================
# LLM Configuration
# =============================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "openai", "ollama", or "none"

# =============================================================================
# MongoDB Configuration
# =============================================================================

MONGODB_URL = os.getenv("MONGODB_URL","mongodb://username:password@localhost:27017/mydatabase")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "pdf_rag_db")

# =============================================================================
# API Data Models
# =============================================================================

class SearchResponse(BaseModel):
    query: str
    search_type: str
    total_results: int
    latency_ms: float
    results: List[Dict[str, Any]]
    tables: List[Dict[str, Any]] = []
    images: List[Dict[str, Any]] = []
    ai_answer: Optional[str] = None
    ai_sources: List[Dict[str, Any]] = []
    ai_provider: Optional[str] = None

class UploadResponse(BaseModel):
    success: bool
    document_id: str
    filename: str
    original_filename: str
    page_count: int
    chunks_created: int
    tables_extracted: int
    images_extracted: int
    processing_time: float

class QuestionRequest(BaseModel):
    question: str
    document_id: Optional[str] = None
    num_context_chunks: int = 10
    search_type: str = "hybrid"

class QuestionResponse(BaseModel):
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    llm_provider: str
    latency_ms: float
