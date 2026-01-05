import os
import re
import json
import shutil
import fitz  # PyMuPDF
import pdfplumber
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from config import (
    DOCUMENTS_DIR, INDEX_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE
)
#docx store
class PersistentDocumentStore:
    """Document store with JSON file persistence"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.documents: Dict[str, Dict] = {}
        self.tables: Dict[str, List[Dict]] = {}
        self.images: Dict[str, List[Dict]] = {}
        self._load_all()
    
    def _load_all(self):
        """Load all documents from disk"""
        docs_file = self.data_dir / "documents_index.json"
        if docs_file.exists():
            with open(docs_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.documents = data.get("documents", {})
                self.tables = data.get("tables", {})
                self.images = data.get("images", {})
    
    def _save_all(self):
        """Save all documents to disk"""
        docs_file = self.data_dir / "documents_index.json"
        data = {
            "documents": self.documents,
            "tables": self.tables,
            "images": self.images
        }
        with open(docs_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def add_document(self, doc_id: str, doc_data: Dict):
        self.documents[doc_id] = doc_data
        self._save_all()
    
    def add_tables(self, doc_id: str, tables: List[Dict]):
        self.tables[doc_id] = tables
        self._save_all()
    
    def add_images(self, doc_id: str, images: List[Dict]):
        self.images[doc_id] = images
        self._save_all()
    
    def get_document(self, doc_id: str) -> Optional[Dict]:
        return self.documents.get(doc_id)
    
    def get_all_documents(self) -> List[Dict]:
        return list(self.documents.values())
    
    def get_all_tables(self) -> List[Dict]:
        all_tables = []
        for doc_id, tables in self.tables.items():
            for table in tables:
                table['document_id'] = doc_id
                all_tables.append(table)
        return all_tables
    
    def delete_document(self, doc_id: str):
        self.documents.pop(doc_id, None)
        self.tables.pop(doc_id, None)
        self.images.pop(doc_id, None)
        self._save_all()
    
    def get_stats(self) -> Dict:
        total_tables = sum(len(t) for t in self.tables.values())
        total_images = sum(len(i) for i in self.images.values())
        return {
            "total_documents": len(self.documents),
            "total_tables": total_tables,
            "total_images": total_images
        }
#txt process
def extract_title_from_text(text: str, filename: str) -> str:
    """Extract a meaningful title from document text"""
    lines = text.strip().split('\n')
    
    for line in lines[:10]:
        line = line.strip()
        if len(line) < 5:
            continue
        if re.match(r'^[\d\s\-\/\.]+$', line):
            continue
        if line.isupper() and len(line) < 10:
            continue
        if len(line) > 5 and len(line) < 100:
            return line
    
    return Path(filename).stem


def sanitize_filename(name: str) -> str:
    """Create a safe filename from a title"""
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    name = re.sub(r'\s+', '_', name)
    name = name.strip('._')
    if len(name) > 50:
        name = name[:50]
    return name or "document"


def smart_chunk_text(text: str, chunk_size: int = CHUNK_SIZE, 
                     overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    """Chunking with sentence awareness"""
    if not text or len(text.strip()) < MIN_CHUNK_SIZE:
        return []
    
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(sentence_pattern, text)
    
    chunks = []
    current_chunk = ""
    current_sentences = []
    chunk_id = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        if len(current_chunk) + len(sentence) + 1 > chunk_size:
            if current_chunk:
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": current_chunk.strip(),
                    "char_count": len(current_chunk.strip()),
                    "sentence_count": len(current_sentences)
                })
                chunk_id += 1
                
                overlap_text = ""
                overlap_sentences = []
                for s in reversed(current_sentences):
                    if len(overlap_text) + len(s) < overlap:
                        overlap_text = s + " " + overlap_text
                        overlap_sentences.insert(0, s)
                    else:
                        break
                
                current_chunk = overlap_text
                current_sentences = overlap_sentences
        
        current_chunk += sentence + " "
        current_sentences.append(sentence)
    
    if current_chunk.strip() and len(current_chunk.strip()) >= MIN_CHUNK_SIZE:
        chunks.append({
            "chunk_id": chunk_id,
            "text": current_chunk.strip(),
            "char_count": len(current_chunk.strip()),
            "sentence_count": len(current_sentences)
        })
    
    return chunks


def extract_sections(text: str) -> List[Dict]:
    """Detect sections/headings in text"""
    sections = []
    lines = text.split('\n')
    
    heading_patterns = [
        r'^#{1,6}\s+(.+)$',
        r'^([A-Z][A-Z\s]{2,50})$',
        r'^(\d+\.?\s+[A-Z].{5,50})$',
        r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,5}):?\s*$',
    ]
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        for pattern in heading_patterns:
            match = re.match(pattern, line)
            if match:
                sections.append({
                    "title": match.group(1) if match.groups() else line,
                    "line_number": i,
                    "text": line
                })
                break
    
    return sections
#Extractions func
def extract_text_with_pymupdf(pdf_path: str) -> Tuple[List[Dict], Dict]:
    """Extract text and metadata from PDF"""
    pages = []
    metadata = {}
    
    doc = fitz.open(pdf_path)
    
    metadata = {
        "title": doc.metadata.get("title", ""),
        "author": doc.metadata.get("author", ""),
        "subject": doc.metadata.get("subject", ""),
        "keywords": doc.metadata.get("keywords", ""),
        "page_count": len(doc)
    }
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        sections = extract_sections(text)
        
        pages.append({
            "page_number": page_num + 1,
            "text": text,
            "char_count": len(text),
            "sections": sections
        })
    
    doc.close()
    return pages, metadata


def extract_tables_with_pdfplumber(pdf_path: str) -> List[Dict]:
    """Extract tables from PDF with accurate header detection"""
    tables = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Try different table extraction settings for better accuracy
                table_settings = {
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                    "snap_tolerance": 3,
                    "join_tolerance": 3,
                }
                
                page_tables = page.extract_tables(table_settings)
                if not page_tables:
                    table_settings = {
                        "vertical_strategy": "text",
                        "horizontal_strategy": "text",
                    }
                    page_tables = page.extract_tables(table_settings)
                for table_idx, table in enumerate(page_tables):
                    if table and len(table) > 1:
                        cleaned_table = []
                        for row in table:
                            if row:
                                cleaned_row = []
                                for cell in row:
                                    if cell:
                                        cell_str = str(cell).strip()
                                        cell_str = re.sub(r'\s+', ' ', cell_str)
                                        cleaned_row.append(cell_str)
                                    else:
                                        cleaned_row.append("")
                                if any(c for c in cleaned_row):
                                    cleaned_table.append(cleaned_row)
                        if len(cleaned_table) < 2:
                            continue
                        header_row_idx = 0
                        headers = []
                        
                        for idx, row in enumerate(cleaned_table):
                            non_empty = [c for c in row if c]
                            if len(non_empty) >= len(row) * 0.5:
                                numeric_cells = sum(1 for c in non_empty if re.match(r'^[\d.,\-+%$]+$', c))
                                if numeric_cells < len(non_empty) * 0.8: 
                                    headers = row
                                    header_row_idx = idx
                                    break
                        
    
                        if not headers:
                            headers = cleaned_table[0]
                            header_row_idx = 0
                        
                        final_headers = []
                        for i, h in enumerate(headers):
                            if h and h.strip():
                                final_headers.append(h.strip())
                            else:
                                final_headers.append(f"Column_{i+1}")
                        
                        rows = []
                        for row in cleaned_table[header_row_idx + 1:]:
                            row_dict = {}
                            has_content = False
                            for i, cell in enumerate(row):
                                if i < len(final_headers):
                                    row_dict[final_headers[i]] = cell if cell else ""
                                    if cell:
                                        has_content = True
                            if has_content:
                                rows.append(row_dict)
                        
                        if rows:
                            tables.append({
                                "table_id": f"table_{page_num}_{table_idx}",
                                "page": page_num + 1,
                                "headers": final_headers,
                                "rows": rows,
                                "row_count": len(rows)
                            })
                            
    except Exception as e:
        print(f"Table extraction error: {e}")
    
    return tables

def extract_images_with_pymupdf(pdf_path: str) -> List[Dict]:
    """Extract image metadata from PDF"""
    images = []
    doc = fitz.open(pdf_path)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images()
        
        for img_idx, img in enumerate(image_list):
            images.append({
                "image_id": f"img_{page_num}_{img_idx}",
                "page": page_num + 1,
                "xref": img[0],
                "width": img[2],
                "height": img[3]
            })
    
    doc.close()
    return images

#pipelines
def process_pdf(pdf_path: str, original_filename: str, search_engine, store, vector_store=None) -> Dict:
    """Process PDF with all extraction and indexing"""
    
    pages, metadata = extract_text_with_pymupdf(pdf_path)
    tables = extract_tables_with_pdfplumber(pdf_path)
    images = extract_images_with_pymupdf(pdf_path)
    
    full_text = "\n\n".join([p["text"] for p in pages])
    
    extracted_title = metadata.get("title") or extract_title_from_text(full_text, original_filename)
    safe_filename = sanitize_filename(extracted_title)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    doc_id = f"{safe_filename}_{timestamp}"
    #Saving files
    doc_folder = DOCUMENTS_DIR / doc_id
    doc_folder.mkdir(parents=True, exist_ok=True)

    new_pdf_name = f"{safe_filename}.pdf"
    new_pdf_path = doc_folder / new_pdf_name
    shutil.copy2(pdf_path, new_pdf_path)
    
    #Chunkings
    all_chunks = []
    for page in pages:
        page_chunks = smart_chunk_text(page["text"])
        current_section = None
        
        for section in page.get("sections", []):
            current_section = section.get("title")
        
        for chunk in page_chunks:
            chunk["page"] = page["page_number"]
            chunk["section"] = current_section
            chunk["document_id"] = doc_id
            chunk["filename"] = new_pdf_name
            all_chunks.append(chunk)
    
    # search engine
    for chunk in all_chunks:
        chunk_id = f"{doc_id}_chunk_{chunk['chunk_id']}"
        search_engine.add_document(
            chunk_id=chunk_id,
            text=chunk["text"],
            metadata={
                "document_id": doc_id,
                "filename": new_pdf_name,
                "page": chunk["page"],
                "section": chunk.get("section")
            }
        )
    
    # sematic search data from vector
    if vector_store:
        print("Adding to vector store...")
        vector_docs = [
            {
                'chunk_id': f"{doc_id}_chunk_{chunk['chunk_id']}",
                'text': chunk["text"],
                'metadata': {
                    "document_id": doc_id,
                    "filename": new_pdf_name,
                    "page": chunk["page"],
                    "section": chunk.get("section")
                }
            }
            for chunk in all_chunks
        ]
        vector_store.add_documents_batch(vector_docs)
    search_engine.save_index(INDEX_DIR / "search_index.json")

    doc_data = {
        "doc_id": doc_id,
        "filename": new_pdf_name,
        "original_filename": original_filename,
        "title": extracted_title,
        "author": metadata.get("author", ""),
        "page_count": len(pages),
        "chunk_count": len(all_chunks),
        "table_count": len(tables),
        "image_count": len(images),
        "total_characters": len(full_text),
        "document_type": "DIGITAL",
        "folder_path": str(doc_folder),
        "pdf_path": str(new_pdf_path),
        "created_at": datetime.now().isoformat()
    }
    
    store.add_document(doc_id, doc_data)
    store.add_tables(doc_id, tables)
    store.add_images(doc_id, images)

    doc_json_path = doc_folder / "document.json"
    with open(doc_json_path, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": doc_data,
            "chunks": all_chunks,
            "tables": tables,
            "images": images
        }, f, indent=2)
    
    print(f"Processed: {original_filename} -> {new_pdf_name}")
    print(f"  - {len(pages)} pages, {len(all_chunks)} chunks")
    
    return {
        "doc_id": doc_id,
        "filename": new_pdf_name,
        "original_filename": original_filename,
        "page_count": len(pages),
        "chunk_count": len(all_chunks),
        "table_count": len(tables),
        "image_count": len(images)
    }

#table searches

def parse_numeric_condition(query: str) -> tuple:
    """Parse numeric conditions from query like 'more than 90', 'above 80', 'less than 50'"""
    import re
    
    # Patterns for numeric conditions
    patterns = [
        (r'(?:more than|greater than|above|over|>\s*|>=\s*)(\d+(?:\.\d+)?)\s*%?', 'gt'),
        (r'(?:less than|below|under|<\s*|<=\s*)(\d+(?:\.\d+)?)\s*%?', 'lt'),
        (r'(?:equal to|equals|=\s*)(\d+(?:\.\d+)?)\s*%?', 'eq'),
        (r'(?:at least|minimum)?\s*(\d+(?:\.\d+)?)\s*%?\s*(?:or more|and above|plus|\+)', 'gte'),
        (r'(?:at most|maximum)?\s*(\d+(?:\.\d+)?)\s*%?\s*(?:or less|and below)', 'lte'),
        (r'between\s*(\d+(?:\.\d+)?)\s*(?:and|to|-)\s*(\d+(?:\.\d+)?)', 'between'),
    ]
    
    for pattern, condition_type in patterns:
        match = re.search(pattern, query.lower())
        if match:
            if condition_type == 'between':
                return (condition_type, float(match.group(1)), float(match.group(2)))
            return (condition_type, float(match.group(1)), None)
    
    return (None, None, None)


def check_numeric_condition(value: str, condition: tuple) -> bool:
    """Check if a value satisfies the numeric condition"""
    condition_type, threshold, threshold2 = condition
    if condition_type is None:
        return False

    import re
    numeric_match = re.search(r'(\d+(?:\.\d+)?)', str(value))
    if not numeric_match:
        return False
    
    num_value = float(numeric_match.group(1))
    
    if condition_type == 'gt':
        return num_value > threshold
    elif condition_type == 'gte':
        return num_value >= threshold
    elif condition_type == 'lt':
        return num_value < threshold
    elif condition_type == 'lte':
        return num_value <= threshold
    elif condition_type == 'eq':
        return num_value == threshold
    elif condition_type == 'between':
        return threshold <= num_value <= threshold2
    
    return False


def search_tables(query: str, store: PersistentDocumentStore, limit: int = 10) -> List[Dict]:
    """Smart table search with numeric condition support"""
    query_lower = query.lower().strip()
    
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                  'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                  'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                  'can', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 
                  'from', 'as', 'into', 'through', 'and', 'but', 'if', 'or',
                  'so', 'than', 'too', 'very', 'just', 'it', 'its', 'you',
                  'your', 'we', 'our', 'they', 'them', 'their', 'this', 'that',
                  'these', 'those', 'what', 'which', 'who', 'whom', 'how',
                  'when', 'where', 'why', 'all', 'each', 'every', 'both',
                  'few', 'more', 'most', 'other', 'some', 'such', 'no', 'not',
                  'only', 'own', 'same', 'then', 'there', 'here', 'now', 'also'}
  
    query_words = [w for w in query_lower.split() if len(w) > 1 and w not in stop_words]
    results = []
    
    numeric_condition = parse_numeric_condition(query)
    has_numeric_condition = numeric_condition[0] is not None
    
    field_hints = []
    field_patterns = [
        r'(?:scored?|marks?|percentage|percent|grade|result|score)\s+(?:in\s+)?(\w+)?',
        r'(\w+)\s+(?:marks?|score|percentage|percent)',
        r'(?:whose?|with)\s+(\w+)',
    ]
    for pattern in field_patterns:
        matches = re.findall(pattern, query_lower)
        field_hints.extend([m for m in matches if m])
    
    all_tables = store.get_all_tables()
    
    for table in all_tables:
        score = 0
        matching_rows = []
        
        headers = table.get("headers", [])
        headers_lower = [str(h).lower() for h in headers]
        headers_text = " ".join(headers_lower)
        
        for word in query_words:
            if word in headers_text:
                score += 3
        
        rows = table.get("rows", [])
        
        for row in rows:
            row_text = " ".join(str(v).lower() for v in row.values())
            row_matched = False
            
            if query_lower in row_text:
                score += 5
                row_matched = True
            
            for word in query_words:
                if word in row_text:
                    score += 1
                    row_matched = True
        
            if has_numeric_condition:
                for header in headers:
                    cell_value = str(row.get(header, ""))
                    if re.search(r'\d', cell_value):
                        
                        header_lower = header.lower()
                        is_relevant_column = not field_hints or any(hint in header_lower for hint in field_hints)
                        
                        if is_relevant_column and check_numeric_condition(cell_value, numeric_condition):
                            score += 10
                            row_matched = True
                            break
            
            if row_matched and row not in matching_rows:
                matching_rows.append(row)
        
        if score > 0 or matching_rows:
            doc = store.get_document(table.get("document_id", ""))
            results.append({
                "document_id": table.get("document_id", ""),
                "document_name": doc.get("filename", "Unknown") if doc else "Unknown",
                "page": table.get("page", 1),
                "table_id": table.get("table_id", ""),
                "headers": table.get("headers", []),
                "matching_rows": matching_rows[:10] if matching_rows else rows[:10],
                "all_rows": rows[:20],
                "total_rows": len(rows),
                "total_matching": len(matching_rows),
                "score": score
            })
    
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:limit]


def format_table_for_context(table_results: List[Dict]) -> str:
    """Format table results as text context for LLM with clear headers"""
    if not table_results:
        return ""
    
    context_parts = []
    for table in table_results[:3]:
        doc_name = table.get("document_name", "Unknown")
        page = table.get("page", "?")
        headers = table.get("headers", [])
        matching_rows = table.get("matching_rows", [])
        all_rows = table.get("all_rows", [])[:10]
        rows = matching_rows if matching_rows else all_rows
        total_matching = table.get("total_matching", 0)
        total_rows = table.get("total_rows", 0)
        
        table_text = f"\n[TABLE DATA from {doc_name}, Page {page}]\n"
        table_text += f"Column Headers: {', '.join(str(h) for h in headers)}\n"
        
        if total_matching > 0:
            table_text += f"Matching Records: {total_matching} out of {total_rows} total rows\n"
        
        table_text += "\n"
        
        table_text += "| " + " | ".join(str(h) for h in headers) + " |\n"
        table_text += "|" + "|".join(["---"] * len(headers)) + "|\n"
        
        for row in rows[:10]:
            row_values = [str(row.get(h, "")).strip() for h in headers]
            table_text += "| " + " | ".join(row_values) + " |\n"
        
        if len(rows) > 10:
            table_text += f"\n... and {len(rows) - 10} more matching rows\n"
        
        context_parts.append(table_text)
    
    return "\n".join(context_parts)
