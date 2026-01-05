# PDF RAG Application

A simple, self-contained PDF knowledge base with AI-powered Q&A using Retrieval-Augmented Generation (RAG).

## Features

- **PDF Upload & Processing**: Extract text and tables from PDFs
- **Smart Search**: TF-IDF based search with phrase matching, keyword frequency, and position bonuses
- **AI Q&A**: Ask questions about your documents using Ollama (local) or OpenAI
- **Table Extraction**: Automatically extracts and searches tables from PDFs
- **Auto-Rename**: Documents are automatically renamed based on their content
- **Persistent Storage**: All data saved to JSON files

## Project Structure

```
pdf_rag_app/
├── backend/
│   ├── app.py              # Complete backend (FastAPI server)
│   ├── requirements.txt    # Python dependencies
│   └── data/               # Storage (auto-created)
│       ├── documents/      # Uploaded PDFs
│       └── index/          # Search index & document metadata
├── frontend/
│   └── index.html          # Complete frontend (single file)
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Start the Server

```bash
python app.py
```

Server starts at http://localhost:8000

### 3. Open the App

Navigate to http://localhost:8000 in your browser.

## AI Q&A Setup

### Option 1: Ollama (Local, Free)

1. Install Ollama: https://ollama.ai
2. Pull a model:
   ```bash
   ollama pull llama3.2:1b    # Small, fast (1GB RAM)
   ollama pull llama3.2       # Medium (4GB RAM)
   ollama pull llama3         # Large (8GB RAM)
   ```
3. Start Ollama: `ollama serve`

### Option 2: OpenAI

Set your API key in the UI settings tab.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/upload` | POST | Upload PDF file |
| `/api/documents` | GET | List all documents |
| `/api/documents/{id}` | DELETE | Delete a document |
| `/api/search` | GET | Search text and tables |
| `/api/tables/search` | GET | Search tables only |
| `/api/ask` | POST | Ask AI a question |
| `/api/stats` | GET | Get system statistics |

## Tech Stack

- **Backend**: FastAPI, PyMuPDF, pdfplumber
- **Frontend**: Bootstrap 5, Vanilla JS
- **Search**: Custom TF-IDF implementation
- **AI**: Ollama / OpenAI integration
- **Storage**: JSON file-based persistence

## License

MIT
