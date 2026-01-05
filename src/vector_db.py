"""
Vector Database using Sentence Transformers
Semantic search with embeddings stored locally
"""
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import re

from config import INDEX_DIR


class VectorStore:
    """Local vector database using sentence-transformers with exact match priority"""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        print("Loading embedding model...")
        self.model = SentenceTransformer(model_name)
        self.embeddings: np.ndarray = np.array([])
        self.metadata: List[Dict] = []  
        self.index_path = INDEX_DIR / "vector_index.npz"
        self.metadata_path = INDEX_DIR / "vector_metadata.json"
        self._query_cache: Dict[str, np.ndarray] = {}  
        self._load_index()
        print(f"Vector store ready with {len(self.metadata)} chunks")
    
    def _load_index(self):
        """Load existing index from disk"""
        if self.index_path.exists() and self.metadata_path.exists():
            try:
                data = np.load(self.index_path)
                self.embeddings = data['embeddings']
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                print(f"Loaded {len(self.metadata)} vectors from disk")
            except Exception as e:
                print(f"Error loading vector index: {e}")
                self.embeddings = np.array([])
                self.metadata = []
    
    def _save_index(self):
        try:
            if len(self.embeddings) > 0:
                np.savez(self.index_path, embeddings=self.embeddings)
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f)
        except Exception as e:
            print(f"Error saving vector index: {e}")
    
    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Get query embedding with caching"""
        query_key = query.lower().strip()
        if query_key not in self._query_cache:
            self._query_cache[query_key] = self.model.encode([query])[0]
            
            if len(self._query_cache) > 100:
                
                keys = list(self._query_cache.keys())[:50]
                for k in keys:
                    del self._query_cache[k]
        return self._query_cache[query_key]
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract the most important terms from a query for better matching"""
        
        question_words = {'what', 'is', 'are', 'how', 'why', 'when', 'where', 'which', 
                         'who', 'whom', 'whose', 'do', 'does', 'did', 'can', 'could',
                         'would', 'should', 'will', 'define', 'explain', 'describe',
                         'tell', 'me', 'about', 'give', 'show'}
        
        
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'to', 'of', 'in',
            'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
            'and', 'but', 'if', 'or', 'so', 'than', 'too', 'very', 'just',
            'it', 'its', 'you', 'your', 'we', 'our', 'they', 'them', 'their',
            'this', 'that', 'these', 'those', 'what', 'which', 'who', 'whom',
            'how', 'when', 'where', 'why', 'all', 'each', 'every', 'both',
            'few', 'more', 'most', 'other', 'some', 'such', 'no', 'not',
            'only', 'own', 'same', 'any', 'many', 'much', 'here', 'there',
            'now', 'then', 'also', 'about', 'like', 'get', 'got', 'make'
        }
        
        all_stop = stopwords | question_words
        words = query.lower().split()
        key_terms = [w for w in words if w not in all_stop and len(w) > 1]
        return key_terms
    
    def _exact_match_score(self, query: str, text: str) -> Tuple[float, str, int]:
        """Calculate exact match bonus, match type, and word count matched"""
        query_lower = query.lower().strip()
        text_lower = text.lower()
        
        key_terms = self._extract_key_terms(query)
        
        if not key_terms:
            key_terms = [w for w in query_lower.split() if len(w) > 2]
        
        if len(key_terms) >= 2:
            key_phrase = ' '.join(key_terms)
            if key_phrase in text_lower:
                return 1.0, "exact", len(key_terms)
        
        if query_lower in text_lower:
            return 0.95, "exact", len(query_lower.split())
        
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'to', 'of', 'in',
            'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
            'and', 'but', 'if', 'or', 'so', 'than', 'too', 'very', 'just',
            'it', 'its', 'you', 'your', 'we', 'our', 'they', 'them', 'their',
            'this', 'that', 'these', 'those', 'what', 'which', 'who', 'whom',
            'how', 'when', 'where', 'why', 'all', 'each', 'every', 'both',
            'few', 'more', 'most', 'other', 'some', 'such', 'no', 'not',
            'only', 'own', 'same', 'any', 'many', 'much', 'here', 'there',
            'now', 'then', 'also', 'about', 'like', 'get', 'got', 'make'
        }
        
        query_words = key_terms if key_terms else [w for w in query_lower.split() if len(w) > 2]
        
        if not query_words:
            return 0.0, "semantic", 0
        
        word_matches = []
        for w in query_words:
            if w in text_lower:
                count = text_lower.count(w)
                is_prominent = text_lower.startswith(w) or f"\n{w}" in text_lower or f". {w}" in text_lower
                word_matches.append((w, count, is_prominent))
        
        matched_words = len(word_matches)
        word_match_ratio = matched_words / len(query_words) if query_words else 0
        prominent_matches = sum(1 for _, _, p in word_matches if p)
        
        if len(query_words) >= 2:
            matched_consecutive = 0
            for i in range(len(query_words) - 1):
                phrase = f"{query_words[i]} {query_words[i+1]}"
                if phrase in text_lower:
                    matched_consecutive += 1
            all_terms_phrase = ' '.join(query_words)
            if all_terms_phrase in text_lower:
                matched_consecutive += 2 
            if matched_consecutive > 0:
            
                score = 0.85 + (matched_consecutive * 0.05)
                if prominent_matches > 0:
                    score = min(0.98, score + 0.05)
                return score, "phrase", matched_words
        
      
        if matched_words == len(query_words) and len(query_words) > 0:
            score = 0.75
            if prominent_matches > 0:
                score = 0.82  
            return score, "all_words", matched_words
        
        elif word_match_ratio >= 0.7:
            return 0.60 + (word_match_ratio * 0.15), "partial", matched_words
        
        elif word_match_ratio >= 0.5:
            return 0.45 + (word_match_ratio * 0.15), "partial", matched_words
     
        elif matched_words > 0:
            return 0.30 + (word_match_ratio * 0.15), "partial", matched_words
        
        return 0.0, "semantic", 0
    
    def add_document(self, chunk_id: str, text: str, metadata: Dict):
        """Add a document chunk to the vector store"""
       
        embedding = self.model.encode([text])[0]
        
        existing_idx = None
        for i, m in enumerate(self.metadata):
            if m.get('chunk_id') == chunk_id:
                existing_idx = i
                break
        
        if existing_idx is not None:
      
            self.embeddings[existing_idx] = embedding
            self.metadata[existing_idx] = {
                'chunk_id': chunk_id,
                'text': text[:500],
                'full_text': text,
                **metadata
            }
        else:
           
            if len(self.embeddings) == 0:
                self.embeddings = embedding.reshape(1, -1)
            else:
                self.embeddings = np.vstack([self.embeddings, embedding])
            
            self.metadata.append({
                'chunk_id': chunk_id,
                'text': text[:500],
                'full_text': text,
                **metadata
            })
        
        self._save_index()
    
    def add_documents_batch(self, documents: List[Dict]):
        """Add multiple documents at once (faster)"""
        if not documents:
            return
        
        texts = [d['text'] for d in documents]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        new_embeddings = []
        new_metadata = []
        updated_count = 0
        
        for i, doc in enumerate(documents):
            chunk_id = doc['chunk_id']
            
            existing_idx = None
            for j, m in enumerate(self.metadata):
                if m.get('chunk_id') == chunk_id:
                    existing_idx = j
                    break
            
            meta_entry = {
                'chunk_id': chunk_id,
                'text': doc['text'][:500],
                'full_text': doc['text'],
                **doc.get('metadata', {})
            }
            
            if existing_idx is not None:
             
                self.embeddings[existing_idx] = embeddings[i]
                self.metadata[existing_idx] = meta_entry
                updated_count += 1
            else:
               
                new_embeddings.append(embeddings[i])
                new_metadata.append(meta_entry)
        if new_embeddings:
            new_embeddings_array = np.array(new_embeddings)
            if len(self.embeddings) == 0:
                self.embeddings = new_embeddings_array
            else:
                self.embeddings = np.vstack([self.embeddings, new_embeddings_array])
            self.metadata.extend(new_metadata)
        
        self._save_index()
        print(f"Added {len(new_metadata)} new, updated {updated_count} documents in vector store. Total: {len(self.metadata)}")
    
    def search(self, query: str, limit: int = 10, min_score: float = 0.01) -> List[Dict]:
        if len(self.embeddings) == 0:
            return []
        query_embedding = self._get_query_embedding(query)
        similarities = self._cosine_similarity(query_embedding, self.embeddings)
        scored_results = []
        for idx in range(len(self.metadata)):
            meta = self.metadata[idx]
            semantic_score = float(similarities[idx])
            text = meta.get('full_text', meta.get('text', ''))
            exact_bonus, match_type, words_matched = self._exact_match_score(query, text)
            if match_type == "exact":
                final_score = 0.95 + (semantic_score * 0.05)
                relevance = "Exact Match"
            elif match_type == "phrase":
                final_score = exact_bonus + (semantic_score * 0.15)
                relevance = "Phrase Match"
            elif match_type == "all_words":
                final_score = exact_bonus + (semantic_score * 0.20)
                relevance = "All Words"
            elif match_type == "partial":
                final_score = (exact_bonus * 0.5) + (semantic_score * 0.5)
                relevance = f"{words_matched} Words"
            else:
                final_score = semantic_score
                relevance = "Related"
            
            if final_score >= min_score:
                scored_results.append({
                    'idx': idx,
                    'score': final_score,
                    'semantic_score': semantic_score,
                    'match_type': match_type,
                    'relevance': relevance,
                    'words_matched': words_matched
                })
        
        scored_results.sort(key=lambda x: x['score'], reverse=True)
        
        results = []
        for item in scored_results[:limit]:
            idx = item['idx']
            meta = self.metadata[idx]
            results.append({
                'chunk_id': meta.get('chunk_id', ''),
                'text': meta.get('text', ''),
                'full_text': meta.get('full_text', meta.get('text', '')),
                'score': item['score'],
                'semantic_score': item['semantic_score'],
                'document_id': meta.get('document_id', ''),
                'document_name': meta.get('filename', 'Unknown'),
                'page': meta.get('page', 1),
                'section': meta.get('section'),
                'match_type': item['match_type'],
                'relevance': item['relevance']
            })
        
        return results
    
    def _cosine_similarity(self, query: np.ndarray, documents: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between query and all documents"""
        query_norm = query / (np.linalg.norm(query) + 1e-10)
        doc_norms = documents / (np.linalg.norm(documents, axis=1, keepdims=True) + 1e-10)
        return np.dot(doc_norms, query_norm)
    
    def delete_document(self, doc_id: str):
        """Delete all chunks for a document"""
        indices_to_remove = []
        for i, m in enumerate(self.metadata):
            if m.get('document_id') == doc_id:
                indices_to_remove.append(i)
        
        if indices_to_remove:
            self.metadata = [m for i, m in enumerate(self.metadata) if i not in indices_to_remove]
            
            if len(self.embeddings) > 0:
                mask = np.ones(len(self.embeddings), dtype=bool)
                mask[indices_to_remove] = False
                self.embeddings = self.embeddings[mask]
            
            self._save_index()
            print(f"Removed {len(indices_to_remove)} chunks for document {doc_id}")
    
    def get_stats(self) -> Dict:
        """Get vector store statistics"""
        return {
            'total_vectors': len(self.metadata),
            'embedding_dimension': self.embeddings.shape[1] if len(self.embeddings) > 0 else 0,
            'model': 'all-MiniLM-L6-v2',
            'cache_size': len(self._query_cache)
        }
 
class HybridSearchEngine:  
    def __init__(self, vector_store: VectorStore, keyword_engine=None):
        self.vector_store = vector_store
        self.keyword_engine = keyword_engine
    
    def search(self, query: str, limit: int = 10, boost_exact: bool = True) -> List[Dict]:
         
        vector_results = self.vector_store.search(query, limit=limit * 2, min_score=0.01)
         
        keyword_results = []
        if self.keyword_engine:
            keyword_results = self.keyword_engine.smart_search(query, limit=limit)
         
        combined = {}
         
        for r in vector_results:
            chunk_id = r['chunk_id']
            combined[chunk_id] = r
         
        for r in keyword_results:
            chunk_id = r['chunk_id']
            if chunk_id not in combined:
                r['match_type'] = 'keyword'
                combined[chunk_id] = r
            else: 
                combined[chunk_id]['score'] = min(1.0, combined[chunk_id]['score'] * 1.2)
         
        results = sorted(combined.values(), key=lambda x: x.get('score', 0), reverse=True)
        
        return results[:limit]
    
    def search_exact(self, query: str, limit: int = 10) -> List[Dict]:
        """Search prioritizing exact text matches"""
        return self.vector_store.search(query, limit=limit, min_score=0.01)
    
    def get_context_for_question(self, question: str, limit: int = 12) -> List[Dict]:
        """
        Get the best context chunks for answering a question
        Always returns results - uses very low threshold
        Prioritizes getting comprehensive coverage from the most relevant document
        """
        question_lower = question.lower()
         
        is_comprehensive = any(word in question_lower for word in [
            'list', 'all methods', 'all types', 'all the', 'how many',
            'what are the', 'enumerate', 'name all'
        ]) 
        fetch_limit = limit * 4 if is_comprehensive else limit * 3
        results = self.search(question, limit=fetch_limit)
        
        if not results:
            return []
 
        if is_comprehensive and len(results) > 0:
            
            doc_scores = {}
            for i, r in enumerate(results[:10]):  
                doc_id = r.get('document_id', '')
                if not doc_id:
                    continue
                score = r.get('score', 0)
                weighted_score = score * (1.0 - i * 0.05)
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {'count': 0, 'total_score': 0, 'top_score': 0}
                doc_scores[doc_id]['count'] += 1
                doc_scores[doc_id]['total_score'] += weighted_score
                if score > doc_scores[doc_id]['top_score']:
                    doc_scores[doc_id]['top_score'] = score
            
            if doc_scores:
                primary_doc = max(doc_scores.keys(), 
                                key=lambda d: doc_scores[d]['top_score'] * min(doc_scores[d]['count'], 3))
                
                additional_from_primary = []
                for m in self.vector_store.metadata:
                    if m.get('document_id') == primary_doc:
                        
                        chunk_id = m.get('chunk_id', '')
                        if not any(r.get('chunk_id') == chunk_id for r in results):
                            additional_from_primary.append({
                                **m,
                                'score': 0.30,  
                                'match_type': 'document_expansion'
                            })
                
                additional_from_primary.sort(key=lambda x: x.get('page', 0))
                results.extend(additional_from_primary[:8])
        
        
        seen_content = set()
        unique_results = []
        for r in results:
            content_key = r.get('text', '')[:100].lower()
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_results.append(r)
        
        
        unique_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        return unique_results[:limit]
