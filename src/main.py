import os
import time
import shutil
import httpx
import numpy as np
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from pymongo import MongoClient

from config import (
    FRONTEND_DIR, UPLOAD_DIR, INDEX_DIR,
    OLLAMA_BASE_URL, OLLAMA_MODEL, LLM_PROVIDER,
    MONGODB_URL, MONGODB_DATABASE,
    SearchResponse, UploadResponse, QuestionRequest, QuestionResponse
)
from search_ai import TFIDFSearchEngine, LLMService
from documents import (
    PersistentDocumentStore, process_pdf, 
    search_tables, format_table_for_context
)
from vector_db import VectorStore, HybridSearchEngine
from mongodb_search import MongoDBSearchService

try:
    mongo_client = MongoClient(
        MONGODB_URL,
        serverSelectionTimeoutMS=10000,
        connectTimeoutMS=10000,
        tls=True,
        tlsAllowInvalidCertificates=False
    )
    db = mongo_client[MONGODB_DATABASE]
  
    mongo_client.admin.command('ping')
    print("connected to MongoDB!")
    mongodb_connected = True

    mongodb_search = MongoDBSearchService(MONGODB_URL, MONGODB_DATABASE)
    print("MongoDB search service started")

except Exception as e:
    print(f"Failed to connect to MongoDB: {e}")
    print("MongoDB features will be disabled. You can still use the app with local storage.")
    print("To fix this:")
    print("   1. Check MongoDB Atlas Network Access settings")
    print("   2. Add your IP address to the whitelist")
    print("   3. Verify database user credentials")
    print("   4. Test connection with MongoDB Compass first")
    mongo_client = None
    db = None
    mongodb_connected = False
    mongodb_search = None

store = PersistentDocumentStore(INDEX_DIR)
search_engine = TFIDFSearchEngine()
llm_service = LLMService()

vector_store = VectorStore()
hybrid_search = HybridSearchEngine(vector_store, search_engine)

search_engine.load_index(INDEX_DIR / "search_index.json")

app = FastAPI(
    title="PDF Knowledge Base API",
    description="PDF processing with semantic search and AI Q&A",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend HTML"""
    html_path = FRONTEND_DIR / "index.html"
    if html_path.exists():
        with open(html_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>PDF Knowledge Base</h1><p>Frontend not found</p>")

@app.get("/api/status")
async def api_status():
    return {
        "message": "PDF Knowledge Base API",
        "status": "running",
        "features": ["semantic_search", "auto_rename", "persistent_storage"],
        "mongodb": {
            "connected": mongodb_connected,
            "database": MONGODB_DATABASE if mongodb_connected else None
        }
    }


@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    stats = store.get_stats()
    vector_stats = vector_store.get_stats()
    return {
        "documents": stats["total_documents"],
        "chunks": len(search_engine.documents),
        "tables": stats["total_tables"],
        "images": stats["total_images"],
        "vocabulary_size": len(search_engine.vocabulary),
        "vectors": vector_stats["total_vectors"],
        "embedding_model": vector_stats["model"]
    }


@app.post("/api/reindex")
async def reindex_all_documents():
    """Rebuild the vector index from all existing documents"""
    import json
    from pathlib import Path
    
    documents_dir = Path("storage/data/documents")
    reindexed = 0
    total_chunks = 0
    errors = []
    
    vector_store.embeddings = np.array([])
    vector_store.metadata = []
    
     
    for doc_folder in documents_dir.iterdir():
        if not doc_folder.is_dir():
            continue
        
        doc_json = doc_folder / "document.json"
        if not doc_json.exists():
            continue
        
        try:
            with open(doc_json, 'r', encoding='utf-8') as f:
                doc_data = json.load(f)
            
            metadata = doc_data.get("metadata", {})
            chunks = doc_data.get("chunks", [])
            doc_id = metadata.get("doc_id", doc_folder.name)
            filename = metadata.get("filename", "Unknown")
            
      
            vector_docs = []
            for chunk in chunks:
                vector_docs.append({
                    'chunk_id': f"{doc_id}_chunk_{chunk['chunk_id']}",
                    'text': chunk["text"],
                    'metadata': {
                        "document_id": doc_id,
                        "filename": filename,
                        "page": chunk["page"],
                        "section": chunk.get("section")
                    }
                })
            
            if vector_docs:
                vector_store.add_documents_batch(vector_docs)
                total_chunks += len(vector_docs)
                reindexed += 1
                print(f"Reindexed {filename}: {len(vector_docs)} chunks")
        
        except Exception as e:
            errors.append({"folder": doc_folder.name, "error": str(e)})
    
    return {
        "success": True,
        "documents_reindexed": reindexed,
        "total_chunks": total_chunks,
        "errors": errors if errors else None
    }


@app.get("/api/llm/status")
async def llm_status():
    """Check LLM configuration status"""
    status = {
        "provider": LLM_PROVIDER,
        "configured": False,
        "details": {}
    }
    
    if LLM_PROVIDER == "ollama":
        status["details"] = {
            "model": OLLAMA_MODEL,
            "base_url": OLLAMA_BASE_URL
        }
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
                status["configured"] = response.status_code == 200
                if status["configured"]:
                    models = response.json().get("models", [])
                    status["details"]["available_models"] = [m.get("name") for m in models]
        except:
            status["configured"] = False
            status["details"]["error"] = "Ollama not running"
    
    return status

 
@app.post("/api/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a PDF document"""
    start_time = time.time()
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")
    
    temp_path = UPLOAD_DIR / file.filename
    try:
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    try:
        result = process_pdf(str(temp_path), file.filename, search_engine, store, vector_store)
        processing_time = time.time() - start_time
        
        if temp_path.exists():
            os.remove(temp_path)
        
        return UploadResponse(
            success=True,
            document_id=result["doc_id"],
            filename=result["filename"],
            original_filename=result["original_filename"],
            page_count=result["page_count"],
            chunks_created=result["chunk_count"],
            tables_extracted=result["table_count"],
            images_extracted=result["image_count"],
            processing_time=processing_time
        )
    except Exception as e:
        if temp_path.exists():
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")


@app.post("/upload")
async def legacy_upload(file: UploadFile = File(...)):
    """Legacy upload endpoint"""
    return await upload_pdf(file)

 
@app.get("/api/documents")
async def list_documents(limit: int = 50, offset: int = 0):
    """List all documents"""
    docs = store.get_all_documents()
    return {
        "documents": docs[offset:offset+limit],
        "total": len(docs)
    }


@app.get("/api/documents/{doc_id}")
async def get_document(doc_id: str):
    """Get document details"""
    doc = store.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@app.get("/api/documents/{doc_id}/download")
async def download_document(doc_id: str):
    """Download the original PDF"""
    doc = store.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    pdf_path = doc.get("pdf_path")
    if not pdf_path or not Path(pdf_path).exists():
        raise HTTPException(status_code=404, detail="PDF file not found")
    
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=doc.get("filename", "document.pdf")
    )


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document and all associated data"""
    doc = store.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    folder_path = doc.get("folder_path")
    folder_deleted = False
    if folder_path and Path(folder_path).exists():
        try:
            folder = Path(folder_path)
            for file in folder.iterdir():
                try:
                    file.unlink()
                except Exception as e:
                    print(f"Could not delete file {file}: {e}")
            try:
                folder.rmdir()
                folder_deleted = True
            except:
                shutil.rmtree(folder_path, ignore_errors=True)
                folder_deleted = True
        except Exception as e:
            print(f"Error during folder deletion: {e}")
    
   
    chunks_to_remove = [cid for cid in search_engine.documents.keys() if cid.startswith(doc_id)]
    for chunk_id in chunks_to_remove:
        search_engine.documents.pop(chunk_id, None)
        search_engine.doc_metadata.pop(chunk_id, None)
        search_engine.tf_vectors.pop(chunk_id, None)
    
  
    vector_store.delete_document(doc_id)
    
    search_engine.save_index(INDEX_DIR / "search_index.json")
    store.delete_document(doc_id)
    
    return {
        "success": True, 
        "message": f"Document {doc_id} deleted",
        "folder_deleted": folder_deleted
    }


@app.post("/api/documents/reprocess-tables")
async def reprocess_all_tables():
    """Reprocess tables for all documents to fix column headers"""
    from documents import extract_tables_with_pdfplumber
    
    all_docs = store.list_documents()
    processed = 0
    errors = []
    tables_updated = 0
    
    for doc in all_docs:
        doc_id = doc.get("doc_id")
        folder_path = doc.get("folder_path")
        
        if not folder_path:
            continue
            
        pdf_path = None
        folder = Path(folder_path)
        if folder.exists():
            for file in folder.iterdir():
                if file.suffix.lower() == '.pdf':
                    pdf_path = str(file)
                    break
        
        if not pdf_path:
            errors.append(f"No PDF found for document {doc_id}")
            continue
        
        try: 
            new_tables = extract_tables_with_pdfplumber(pdf_path)
             
            for table in new_tables:
                table["document_id"] = doc_id
             
            doc_data = store.get_document(doc_id)
            if doc_data:
                doc_data["tables"] = new_tables
                store.documents[doc_id] = doc_data
                tables_updated += len(new_tables)
            
            processed += 1
            print(f" Reprocessed tables for: {doc.get('filename', doc_id)}")
            
        except Exception as e:
            errors.append(f"Error processing {doc_id}: {str(e)}")
            print(f" Error reprocessing {doc_id}: {e}")

    store.save_index()
    
    return {
        "success": True,
        "documents_processed": processed,
        "tables_updated": tables_updated,
        "errors": errors if errors else None
    }


@app.post("/api/documents/{doc_id}/reprocess-tables")
async def reprocess_document_tables(doc_id: str):
    """Reprocess tables for a specific document to fix column headers"""
    from documents import extract_tables_with_pdfplumber
    
    doc = store.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    folder_path = doc.get("folder_path")
    if not folder_path:
        raise HTTPException(status_code=400, detail="Document folder not found")
   
    pdf_path = None
    folder = Path(folder_path)
    if folder.exists():
        for file in folder.iterdir():
            if file.suffix.lower() == '.pdf':
                pdf_path = str(file)
                break
    
    if not pdf_path:
        raise HTTPException(status_code=400, detail="PDF file not found in document folder")
    
    try:
        new_tables = extract_tables_with_pdfplumber(pdf_path)
        
        
        for table in new_tables:
            table["document_id"] = doc_id
        
        doc["tables"] = new_tables
        store.documents[doc_id] = doc
        store.save_index()
        
        return {
            "success": True,
            "document_id": doc_id,
            "tables_extracted": len(new_tables),
            "table_headers": [t.get("headers", []) for t in new_tables]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reprocess tables: {str(e)}")



@app.get("/api/search", response_model=SearchResponse)
async def search(
    query: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=100),
    search_type: str = Query("auto", description="Search type: auto, semantic, keyword, exact, mongodb"),
    with_ai: bool = Query(True, description="Include AI-generated answer")
):
    """
    Smart search with AI-powered answers
    - Returns search results AND an AI-generated answer at the top
    - auto: Uses vector search with exact match boosting (recommended)
    - exact: Prioritizes exact text matches
    - semantic: Pure semantic/meaning-based search
    - keyword: Traditional keyword search
    - mongodb: Search directly in MongoDB (when available)
    """
    start_time = time.time()

    if search_type == "mongodb" and mongodb_connected and mongodb_search:
        print(f"🔍 Using MongoDB search for query: '{query}'")
        results = mongodb_search.search_chunks(query, limit)
        search_type_used = "mongodb"
    else:
        if search_type == "exact":
            results = hybrid_search.search_exact(query, limit)
        elif search_type == "semantic":
            results = vector_store.search(query, limit, min_score=0.1)
        elif search_type == "keyword":
            results = search_engine.smart_search(query, limit)
        else: 
            results = hybrid_search.search(query, limit)
        search_type_used = search_type

    tables = search_tables(query, store, limit=5)
    
    ai_answer = None
    ai_sources = []
    ai_provider = None
    
    if with_ai and (results or tables):
        try:
            table_context = format_table_for_context(tables)
            query_lower = query.lower()
            topic_words = set()
            for word in query_lower.split():
                if word not in {'what', 'is', 'are', 'how', 'why', 'when', 'where', 'which', 
                               'who', 'the', 'a', 'an', 'to', 'of', 'in', 'for', 'on', 'with',
                               'do', 'does', 'did', 'can', 'could', 'would', 'should', 'will',
                               'define', 'explain', 'describe', 'tell', 'me', 'about'} and len(word) > 2:
                    topic_words.add(word)
      
            def context_relevance(r):
                text = r.get('full_text', r.get('text', '')).lower()
                base_score = r.get('score', 0)
        
                matches = sum(1 for w in topic_words if w in text)
                topic_boost = (matches / max(len(topic_words), 1)) * 0.4
                return base_score + topic_boost
            
            sorted_results = sorted(results, key=context_relevance, reverse=True)
            ai_context = sorted_results[:6] 
            ai_answer, ai_provider = await llm_service.answer_question(query, ai_context, table_context)
      
            seen_sources = set()
            for r in results[:5]:
                source_key = f"{r.get('document_name', 'Unknown')}_{r.get('page', 1)}"
                if source_key not in seen_sources:
                    seen_sources.add(source_key)
                    ai_sources.append({
                        "document_name": r.get("document_name", "Unknown"),
                        "document_id": r.get("document_id", ""),
                        "page": r.get("page", 1),
                        "type": "text"
                    })
            
            for t in tables[:3]:
                source_key = f"{t.get('document_name', 'Unknown')}_{t.get('page', 1)}_table"
                if source_key not in seen_sources:
                    seen_sources.add(source_key)
                    ai_sources.append({
                        "document_name": t.get("document_name", "Unknown"),
                        "document_id": t.get("document_id", ""),
                        "page": t.get("page", 1),
                        "type": "table",
                        "headers": t.get("headers", [])
                    })
                    
        except Exception as e:
            print(f"AI answer generation failed: {e}")
            ai_answer = None

    latency_ms = (time.time() - start_time) * 1000

    return SearchResponse(
        query=query,
        search_type=search_type_used,
        total_results=len(results) + len(tables),
        latency_ms=latency_ms,
        results=results,
        tables=tables,
        images=[],
        ai_answer=ai_answer,
        ai_sources=ai_sources,
        ai_provider=ai_provider
    )


@app.get("/api/tables/search")
async def search_in_tables(query: str = Query(..., min_length=1)):
    """Search within tables"""
    return search_tables(query, store)


@app.get("/search")
async def legacy_search(query: str, limit: int = 10):
    """Legacy search endpoint"""
    start_time = time.time()
    results = search_engine.hybrid_search(query, limit)
    tables = search_tables(query, store)
    latency_ms = (time.time() - start_time) * 1000
    
    return {
        "results": results,
        "tables": tables,
        "summary": f"Found {len(results) + len(tables)} results for '{query}' in {latency_ms:.0f}ms"
    }


@app.post("/api/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question and get an AI-generated answer - fast and accurate"""
    start_time = time.time()
    
    question_lower = request.question.lower()
    is_about_question = any(phrase in question_lower for phrase in 
                           ['about', 'summary', 'summarize', 'overview', 'describe the document',
                            'what is this', 'what are these', 'tell me about'])
    
    
    results = hybrid_search.get_context_for_question(
        request.question, 
        request.num_context_chunks
    )
    
    if request.document_id:
        results = [r for r in results if r.get("document_id") == request.document_id]
    
   
    if not results and is_about_question:
        all_docs = store.get_all_documents()
        if request.document_id:
            all_docs = [d for d in all_docs if d.get('doc_id') == request.document_id]
        
        for doc in all_docs[:2]:
            doc_chunks = []
            for chunk_id, metadata in search_engine.doc_metadata.items():
                if metadata.get('document_id') == doc.get('doc_id'):
                    doc_chunks.append({
                        'chunk_id': chunk_id,
                        'text': search_engine.documents.get(chunk_id, '')[:600],
                        'full_text': search_engine.documents.get(chunk_id, ''),
                        'document_id': doc.get('doc_id'),
                        'document_name': doc.get('filename'),
                        'page': metadata.get('page', 1),
                        'score': 0.5,
                        'match_type': 'fallback'
                    })
            results.extend(sorted(doc_chunks, key=lambda x: x.get('page', 1))[:2])
    
    
    table_results = search_tables(request.question, store, limit=2)
    if request.document_id:
        table_results = [t for t in table_results if t.get("document_id") == request.document_id]
    
    table_context = format_table_for_context(table_results)
    
    
    answer, provider = await llm_service.answer_question(request.question, results, table_context)
    
    latency_ms = (time.time() - start_time) * 1000
    

    sources = [
        {
            "document_name": r.get("document_name", "Unknown"),
            "document_id": r.get("document_id", ""),
            "page": r.get("page", 1),
            "score": r.get("score", 0),
            "excerpt": r.get("text", "")[:200] + "...",
            "type": "text",
            "match_type": r.get("match_type", "semantic")
        }
        for r in results
    ]
    
    for t in table_results:
        sources.append({
            "document_name": t.get("document_name", "Unknown"),
            "document_id": t.get("document_id", ""),
            "page": t.get("page", 1),
            "score": t.get("score", 0) / 10,
            "excerpt": f"Table with {t.get('total_rows', 0)} rows",
            "type": "table",
            "match_type": "table"
        })
    
    return QuestionResponse(
        question=request.question,
        answer=answer,
        sources=sources,
        llm_provider=provider,
        latency_ms=latency_ms
    )


@app.get("/api/ask")
async def ask_question_get(
    question: str = Query(..., min_length=1),
    document_id: Optional[str] = None,
    num_chunks: int = Query(5, ge=1, le=20)
):
    """GET endpoint for asking questions"""
    request = QuestionRequest(
        question=question,
        document_id=document_id,
        num_context_chunks=num_chunks
    )
    return await ask_question(request)


@app.post("/api/vectors/rebuild")
async def rebuild_vector_index():
    """Rebuild vector index from existing keyword index"""
    start_time = time.time()
    
  
    docs_to_add = []
    for chunk_id, text in search_engine.documents.items():
        metadata = search_engine.doc_metadata.get(chunk_id, {})
        docs_to_add.append({
            'chunk_id': chunk_id,
            'text': text,
            'metadata': metadata
        })
    
    if docs_to_add:
        vector_store.add_documents_batch(docs_to_add)
    
    elapsed = time.time() - start_time
    
    return {
        "success": True,
        "vectors_created": len(docs_to_add),
        "time_seconds": round(elapsed, 2)
    }


@app.get("/api/vectors/stats")
async def vector_stats():
    """Get vector store statistics"""
    return vector_store.get_stats()


if __name__ == "__main__":
    import uvicorn
    print("Starting PDF Knowledge Base with Vector Search...")
    print(f"Loaded {len(store.documents)} documents")
    print(f"Search index has {len(search_engine.documents)} chunks")
    print(f"Vector store has {vector_store.get_stats()['total_vectors']} vectors")
    uvicorn.run(app, host="localhost", port=8000)
