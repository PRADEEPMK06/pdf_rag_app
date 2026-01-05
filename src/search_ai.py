"""
Search Engine and AI/LLM Services
"""
import re
import math
import httpx
from typing import List, Dict, Any, Tuple
from collections import defaultdict, Counter
from pathlib import Path
import json

from config import (
    OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL,
    OLLAMA_BASE_URL, OLLAMA_MODEL, LLM_PROVIDER
)

# =============================================================================
# LLM Service
# =============================================================================

class LLMService:
    """LLM service supporting OpenAI and Ollama"""
    
    def __init__(self):
        self.provider = LLM_PROVIDER
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def close(self):
        await self.client.aclose()
    
    def _build_prompt(self, question: str, context_chunks: List[Dict], table_context: str = "") -> str:
        """Build RAG prompt - accurate and concise by default, detailed when asked"""
        
        # Extract key terms from the question for relevance filtering
        question_lower = question.lower()
        question_words = set()
        for word in question_lower.split():
            # Skip question words and stop words
            if word not in {'what', 'is', 'are', 'how', 'why', 'when', 'where', 'which', 
                           'who', 'the', 'a', 'an', 'to', 'of', 'in', 'for', 'on', 'with',
                           'do', 'does', 'did', 'can', 'could', 'would', 'should', 'will',
                           'define', 'explain', 'describe', 'tell', 'me', 'about'} and len(word) > 2:
                question_words.add(word)
        
        # Score and sort chunks by relevance to question
        def chunk_relevance(chunk):
            text = chunk.get('full_text', chunk.get('text', '')).lower()
            score = chunk.get('score', 0)
            
            # Boost chunks that contain the key terms
            term_matches = sum(1 for w in question_words if w in text)
            term_boost = term_matches / max(len(question_words), 1) * 0.3
            
            # Boost chunks that have key terms in prominent positions
            prominence_boost = 0
            for w in question_words:
                if text.startswith(w) or f"\n{w}" in text or f". {w}" in text:
                    prominence_boost += 0.1
            
            return score + term_boost + prominence_boost
        
        # Sort chunks by relevance
        sorted_chunks = sorted(context_chunks, key=chunk_relevance, reverse=True)
        
        # Build context - prioritize most relevant chunks
        context_parts = []
        for c in sorted_chunks:
            source = f"[{c.get('document_name', 'Unknown')}, Page {c.get('page', '?')}]"
            text = c.get('full_text', c.get('text', ''))
            context_parts.append(f"{source}\n{text}")
        
        context_text = "\n\n---\n\n".join(context_parts)
        
        full_context = context_text
        if table_context:
            full_context += f"\n\n--- TABLE DATA ---\n{table_context}"
        
        # Extract the exact term the user is asking about
        question_lower = question.lower()
        
        # CRITICAL: Detect method name queries and ensure exact matching
        # This prevents confusion between similar methods like headSet vs tailSet
        method_terms = ['headset', 'tailset', 'subset', 'method', 'function']
        is_method_query = any(term in question_lower for term in method_terms)
        
        # Detect table/data queries (marks, students, scores, records, etc.)
        table_query_terms = ['student', 'score', 'mark', 'grade', 'percent', 'above', 'below', 
                            'more than', 'less than', 'greater', 'who scored', 'top', 'result',
                            'record', 'data', 'list all', 'show all', 'find all']
        is_table_query = any(term in question_lower for term in table_query_terms)
        
        # Detect question intent
        question_lower = question.lower()
        
        # Check for detailed explanation requests
        wants_detail = any(word in question_lower for word in [
            'explain', 'describe', 'elaborate', 'in detail', 'detailed',
            'tell me more', 'how does', 'why does', 'what exactly', 'thoroughly'
        ])
        
        # Check for list requests
        is_list_question = any(word in question_lower for word in [
            'list', 'what are', 'name all', 'enumerate', 'give me all', 'types of',
            'all methods', 'all the', 'methods of', 'how many'
        ])
        
        # Check for definition/simple questions
        is_definition = any(word in question_lower for word in [
            'what is', 'define', 'meaning of', 'what does'
        ]) and not wants_detail
        
        # Check for example requests
        wants_examples = any(word in question_lower for word in [
            'example', 'show me', 'demonstrate', 'code', 'sample'
        ])
        
        # Special instructions for table/data queries
        if is_table_query and table_context:
            instructions = """INSTRUCTIONS FOR TABLE DATA QUERIES:
1. Present the data in a CLEAR TABLE FORMAT with proper headers
2. Show ALL matching rows from the TABLE DATA section
3. Use the EXACT column headers from the table
4. Format as a markdown table with | separators
5. Include a summary count (e.g., "Found X students/records matching the criteria")
6. If searching for specific values (like names), show the COMPLETE row with all columns
7. For numeric queries (like "above 90%"), show only rows that match the condition
8. Always include the header row first, then the data rows"""
            
        elif wants_detail:
            instructions = """INSTRUCTIONS:
1. Provide a DETAILED and comprehensive answer using the context
2. Explain thoroughly with all relevant information
3. Use clear formatting with sections if needed
4. Include examples if they exist in the context
5. Use ONLY information from the context provided"""
            
        elif is_definition:
            instructions = """INSTRUCTIONS:
1. Give a DIRECT 1-2 sentence answer from the context
2. Be precise and accurate
3. Do NOT elaborate unless asked
4. Use ONLY the context provided"""
            
        elif is_list_question:
            instructions = """INSTRUCTIONS:
1. Provide a COMPLETE bullet-point list ONLY from the context
2. Include ALL items mentioned in any part of the context - do not leave any out
3. One line per item with brief description if available
4. ONLY include items explicitly mentioned in the provided context
5. If items are mentioned multiple times, list them once
6. DO NOT invent or add any items not explicitly stated in the context
7. STOP after listing items from the context - do not continue with unrelated content"""
            
        elif wants_examples:
            instructions = """INSTRUCTIONS:
1. Show examples from the context
2. Keep explanations brief
3. Include code if in context
4. Use ONLY the context provided"""
            
        else:
            # Default: Short, accurate, direct answer
            instructions = """INSTRUCTIONS:
1. Answer in 1-3 sentences
2. Be DIRECT and ACCURATE - answer EXACTLY what was asked
3. Focus on the SPECIFIC TOPIC/TERMS in the question
4. If asked "what is X", define X - not something else
5. Do NOT pick a random topic from the context - answer the QUESTION
6. Use ONLY the context provided
7. If the question asks about a general concept, explain that concept
8. Do NOT jump to specific examples unless asked"""
        
        # CRITICAL: Detect when user asks about a GENERAL concept vs specific method
        # This prevents AI from answering about "headSet" when asked about "navigation methods"
        general_concept_indicators = ['what is', 'what are', 'define', 'meaning of', 'explain']
        is_general_question = any(ind in question_lower for ind in general_concept_indicators)
        
        # Check if user is asking about a specific method name
        specific_methods = ['headset', 'tailset', 'subset', 'ceiling', 'floor', 'higher', 'lower']
        asking_specific_method = any(m in question_lower for m in specific_methods)
        
        if is_general_question and not asking_specific_method:
            # User wants general explanation, NOT specific method details
            instructions += """\n\nIMPORTANT - GENERAL QUESTION DETECTED:
- The user is asking about a GENERAL CONCEPT, not a specific method
- Provide an OVERVIEW or DEFINITION of the concept
- Do NOT focus on one specific method/example unless asked
- If asked "what are navigation methods", explain what navigation methods ARE in general
- List the types/categories if relevant, but don't deep-dive into one specific method"""
        
        # Add special instructions for method/function queries ONLY if asking about specific method
        if is_method_query and asking_specific_method:
            # Check which specific method the user is asking about
            if 'headset' in question_lower and 'tailset' not in question_lower:
                instructions += """\n\nCRITICAL ACCURACY REQUIREMENT:
- The user is asking about 'headSet' (NOT tailSet)
- headSet() returns elements BEFORE the specified element
- DO NOT confuse this with tailSet() which returns elements AFTER
- Be precise about which method you are describing"""
            elif 'tailset' in question_lower and 'headset' not in question_lower:
                instructions += """\n\nCRITICAL ACCURACY REQUIREMENT:
- The user is asking about 'tailSet' (NOT headSet)
- tailSet() returns elements AFTER the specified element
- DO NOT confuse this with headSet() which returns elements BEFORE
- Be precise about which method you are describing"""
            else:
                instructions += """\n\nCRITICAL ACCURACY REQUIREMENT:
- Pay close attention to the EXACT method/function name the user is asking about
- DO NOT confuse similar method names (e.g., headSet vs tailSet)
- Only describe the specific method that matches the user's query
- Quote the exact method name from the context"""
        
        # Extract the main topic/subject of the question
        topic_hint = ""
        if question_words:
            topic_hint = f"\nTOPIC TO ANSWER: {', '.join(sorted(question_words)[:5])}"
        
        # Build clearer prompt with explicit topic focus
        prompt = f"""Answer the following question using ONLY the provided context.

{instructions}
{topic_hint}

REMEMBER: Answer the question "{question}" - not something else from the context.

CONTEXT:
{full_context}

QUESTION: {question}

ANSWER (be specific to the question):

ANSWER:"""
        return prompt
    
    async def ask_openai(self, question: str, context_chunks: List[Dict], table_context: str = "") -> str:
        """Query OpenAI API"""
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set")
        
        prompt = self._build_prompt(question, context_chunks, table_context)
        
        response = await self.client.post(
            f"{OPENAI_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": OPENAI_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that provides ACCURATE answers based on document context. CRITICAL: When asked about a specific method or function, describe ONLY that exact method - do NOT confuse similar names like headSet vs tailSet. Always match the exact term the user asked about."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 2000
            }
        )
        
        if response.status_code != 200:
            raise Exception(f"OpenAI API error: {response.status_code} - {response.text}")
        
        data = response.json()
        return data["choices"][0]["message"]["content"]
    
    async def ask_ollama(self, question: str, context_chunks: List[Dict], table_context: str = "") -> str:
        """Query Ollama API with optimized settings"""
        # Detect if user wants a list (needs more chunks)
        question_lower = question.lower()
        is_list = any(word in question_lower for word in [
            'list', 'all methods', 'all types', 'name all', 'enumerate', 
            'how many', 'what are the', 'types of', 'methods of', 'all the'
        ])
        
        # Use more chunks for list questions - they need comprehensive coverage
        max_chunks = 10 if is_list else 5
        
        # Extract key topic words from question for filtering
        topic_words = set()
        for word in question_lower.split():
            if word not in {'what', 'is', 'are', 'how', 'why', 'when', 'where', 'which', 
                           'who', 'the', 'a', 'an', 'to', 'of', 'in', 'for', 'on', 'with',
                           'do', 'does', 'did', 'can', 'could', 'would', 'should', 'will',
                           'define', 'explain', 'describe', 'tell', 'me', 'about'} and len(word) > 2:
                topic_words.add(word)
        
        # Score chunks by topic relevance, not just search score
        def topic_score(chunk):
            text = chunk.get('full_text', chunk.get('text', '')).lower()
            base_score = chunk.get('score', 0)
            
            # Count how many topic words appear in the chunk
            topic_matches = sum(1 for w in topic_words if w in text)
            topic_ratio = topic_matches / max(len(topic_words), 1)
            
            # Big boost for chunks with all or most topic words
            if topic_ratio >= 0.8:
                return base_score + 0.5
            elif topic_ratio >= 0.5:
                return base_score + 0.3
            elif topic_ratio > 0:
                return base_score + (topic_ratio * 0.2)
            return base_score
        
        # Sort by topic relevance
        sorted_chunks = sorted(context_chunks, key=topic_score, reverse=True)
        limited_chunks = sorted_chunks[:max_chunks]
        
        # Truncate text for memory efficiency but keep more content
        for chunk in limited_chunks:
            if 'full_text' in chunk:
                chunk['full_text'] = chunk['full_text'][:1200]  # More content for better coverage
        
        if table_context and len(table_context) > 1000:
            table_context = table_context[:1000] + "..."
        
        prompt = self._build_prompt(question, limited_chunks, table_context)
        
        # Detect if user wants detailed response
        question_lower = question.lower()
        wants_detail = any(word in question_lower for word in [
            'explain', 'describe', 'elaborate', 'detail', 'in detail', 
            'tell me more', 'how does', 'why does', 'what exactly', 'thoroughly'
        ])
        
        # Also check for list questions - they need more tokens
        is_list = any(word in question_lower for word in [
            'list', 'what are', 'all methods', 'all types', 'name all', 'enumerate'
        ])
        
        # Token limits based on question type - list questions need more output
        max_tokens = 800 if is_list else (500 if wants_detail else 250)
        
        # Build a focused system instruction - be very explicit
        topic_str = ', '.join(sorted(topic_words)[:4]) if topic_words else 'the topic'
        system_instruction = f"""You answer questions accurately. 
CRITICAL: The user asked about "{topic_str}". 
Answer ONLY about {topic_str}. 
Do NOT answer about other topics from the context.
If asked "what is X", define X specifically."""
        
        try:
            response = await self.client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "system": system_instruction,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Low = more focused/accurate
                        "num_predict": max_tokens,
                        "num_ctx": 4096,  # Context window
                        "top_p": 0.9,
                        "repeat_penalty": 1.15
                    }
                },
                timeout=120.0
            )
            
            if response.status_code != 200:
                error_text = response.text
                try:
                    error_data = response.json()
                    error_text = error_data.get("error", error_text)
                except:
                    pass
                raise Exception(f"Ollama error: {error_text}")
            
            data = response.json()
            answer = data.get("response", "").strip()
            if not answer:
                raise Exception("Ollama returned empty response")
            return answer
        except httpx.ConnectError:
            raise Exception("Cannot connect to Ollama. Make sure Ollama is running (ollama serve)")
        except httpx.TimeoutException:
            raise Exception("Ollama request timed out. Try a shorter question or smaller model.")
    
    async def answer_question(self, question: str, context_chunks: List[Dict], table_context: str = "") -> Tuple[str, str]:
        """Answer question using configured LLM provider"""
        if not context_chunks and not table_context:
            return "No relevant documents found to answer this question.", self.provider
        
        try:
            if self.provider == "openai":
                answer = await self.ask_openai(question, context_chunks, table_context)
            elif self.provider == "ollama":
                answer = await self.ask_ollama(question, context_chunks, table_context)
            else:
                answer = self._fallback_answer(question, context_chunks)
            
            return answer, self.provider
        except Exception as e:
            return f"Error getting LLM response: {str(e)}", self.provider
    
    def _fallback_answer(self, question: str, context_chunks: List[Dict]) -> str:
        """Fallback when no LLM is available"""
        response = f"**No LLM configured.** Here are the most relevant excerpts:\n\n"
        
        for i, chunk in enumerate(context_chunks[:3], 1):
            doc_name = chunk.get('document_name', 'Unknown')
            page = chunk.get('page', '?')
            text = chunk.get('text', '')[:500]
            response += f"**{i}. {doc_name} (Page {page})**\n{text}...\n\n"
        
        response += "\n*Configure an LLM (OpenAI or Ollama) for AI-generated answers.*"
        return response


# =============================================================================
# TF-IDF Search Engine with BM25
# =============================================================================

class TFIDFSearchEngine:
    """TF-IDF based semantic search with BM25 ranking"""
    
    def __init__(self):
        self.documents: Dict[str, str] = {}  # chunk_id -> text
        self.doc_metadata: Dict[str, Dict] = {}  # chunk_id -> metadata
        self.vocabulary: Dict[str, int] = {}  # word -> index
        self.idf: Dict[str, float] = {}  # word -> idf score
        self.tf_vectors: Dict[str, Dict[str, float]] = {}  # chunk_id -> {word: tf}
        self.word_to_chunks: Dict[str, set] = defaultdict(set)  # word -> set of chunk_ids
        
        # Comprehensive stop words list
        self.stop_words = {
            # Articles & Determiners
            'the', 'a', 'an', 'this', 'that', 'these', 'those',
            # Be verbs
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'am',
            # Have verbs
            'have', 'has', 'had', 'having',
            # Do verbs
            'do', 'does', 'did', 'doing',
            # Modal verbs
            'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can',
            # Prepositions
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
            'through', 'during', 'before', 'after', 'above', 'below', 'between',
            'under', 'again', 'further', 'then', 'once', 'about', 'against', 'over',
            # Conjunctions
            'and', 'but', 'if', 'or', 'because', 'until', 'while', 'nor', 'yet', 'so',
            # Pronouns
            'it', 'its', 'you', 'your', 'yours', 'we', 'our', 'ours', 'they', 'them',
            'their', 'theirs', 'he', 'she', 'him', 'her', 'his', 'hers', 'me', 'my', 'mine',
            # Question words (keep these for context but low weight)
            'what', 'which', 'who', 'whom', 'whose', 'where', 'when', 'why', 'how',
            # Common adverbs
            'very', 'just', 'too', 'also', 'only', 'now', 'here', 'there', 'than',
            # Other common words
            'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'not', 'own', 'same', 'any', 'many', 'much',
            # Filler words
            'like', 'get', 'got', 'make', 'made', 'take', 'took', 'come', 'came',
            'give', 'gave', 'go', 'went', 'see', 'saw', 'know', 'knew',
        }
    
    def _tokenize(self, text: str, keep_original: bool = False) -> List[str]:
        """Tokenize text into words with optional stemming"""
        text = text.lower()
        words = re.findall(r'\b[a-z0-9]{2,}\b', text)
        
        filtered = [w for w in words if w not in self.stop_words and len(w) > 1]
        
        if keep_original:
            return filtered
        
        return [self._simple_stem(w) for w in filtered]
    
    def _simple_stem(self, word: str) -> str:
        """Simple suffix-stripping stemmer"""
        if len(word) <= 3:
            return word
        suffixes = ['ation', 'ment', 'ness', 'able', 'ible', 'ful', 'less', 
                    'ive', 'ing', 'ed', 'er', 'est', 'ly', 'es', 's']
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                return word[:-len(suffix)]
        return word
    
    def add_document(self, chunk_id: str, text: str, metadata: Dict):
        """Add a document chunk to the index"""
        self.documents[chunk_id] = text
        self.doc_metadata[chunk_id] = metadata
        
        words = self._tokenize(text)
        word_counts = Counter(words)
        total_words = len(words)
        
        if total_words == 0:
            return
        
        tf = {}
        for word, count in word_counts.items():
            tf[word] = count / total_words
            self.word_to_chunks[word].add(chunk_id)
            if word not in self.vocabulary:
                self.vocabulary[word] = len(self.vocabulary)
        
        self.tf_vectors[chunk_id] = tf
    
    def _rebuild_idf(self):
        """Rebuild IDF scores"""
        num_docs = len(self.documents)
        if num_docs == 0:
            return
        
        for word, chunk_ids in self.word_to_chunks.items():
            df = len(chunk_ids)
            self.idf[word] = math.log(num_docs / df) + 1
    
    def _generate_highlights(self, text: str, query_words: List[str], context: int = 80) -> List[str]:
        """Generate text snippets with query words highlighted"""
        highlights = []
        text_lower = text.lower()
        used_ranges = []
        
        sorted_words = sorted(query_words, key=len, reverse=True)
        
        for word in sorted_words[:5]:
            if len(word) < 2:
                continue
            idx = text_lower.find(word)
            while idx != -1 and len(highlights) < 3:
                overlaps = any(start <= idx <= end or start <= idx + len(word) <= end 
                              for start, end in used_ranges)
                if not overlaps:
                    start = max(0, idx - context)
                    end = min(len(text), idx + len(word) + context)
                    
                    while start > 0 and text[start] not in ' \n\t.':
                        start -= 1
                    while end < len(text) and text[end] not in ' \n\t.':
                        end += 1
                    
                    snippet = text[start:end].strip()
                    if start > 0:
                        snippet = "..." + snippet
                    if end < len(text):
                        snippet = snippet + "..."
                    
                    highlights.append(snippet)
                    used_ranges.append((start, end))
                
                idx = text_lower.find(word, idx + 1)
        
        return highlights[:3]
    
    def search(self, query: str, limit: int = 10) -> List[Dict]:
        """Search using TF-IDF similarity"""
        if not self.documents:
            return []
        
        self._rebuild_idf()
        
        query_words = self._tokenize(query)
        if not query_words:
            return []
        
        query_counts = Counter(query_words)
        query_total = len(query_words)
        query_tf = {w: c/query_total for w, c in query_counts.items()}
        
        scores = []
        
        for chunk_id, doc_tf in self.tf_vectors.items():
            score = 0
            for word in query_words:
                if word in doc_tf and word in self.idf:
                    tf_idf_doc = doc_tf[word] * self.idf[word]
                    tf_idf_query = query_tf[word] * self.idf.get(word, 1)
                    score += tf_idf_doc * tf_idf_query
            
            if score > 0:
                scores.append((chunk_id, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for chunk_id, score in scores[:limit]:
            metadata = self.doc_metadata.get(chunk_id, {})
            text = self.documents.get(chunk_id, "")
            highlights = self._generate_highlights(text, query_words)
            
            results.append({
                "chunk_id": chunk_id,
                "text": text[:500] + "..." if len(text) > 500 else text,
                "full_text": text,
                "score": min(score * 5, 1.0),
                "document_id": metadata.get("document_id", ""),
                "document_name": metadata.get("filename", "Unknown"),
                "page": metadata.get("page", 1),
                "section": metadata.get("section"),
                "highlights": highlights
            })
        
        return results
    
    def keyword_search(self, query: str, limit: int = 10) -> List[Dict]:
        """Simple keyword matching search"""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        results = []
        
        for chunk_id, text in self.documents.items():
            text_lower = text.lower()
            score = 0
            
            for word in query_words:
                if word in text_lower:
                    score += text_lower.count(word)
            
            if score > 0:
                metadata = self.doc_metadata.get(chunk_id, {})
                highlights = self._generate_highlights(text, list(query_words))
                
                results.append({
                    "chunk_id": chunk_id,
                    "text": text[:500] + "..." if len(text) > 500 else text,
                    "full_text": text,
                    "score": min(score / 10, 1.0),
                    "document_id": metadata.get("document_id", ""),
                    "document_name": metadata.get("filename", "Unknown"),
                    "page": metadata.get("page", 1),
                    "section": metadata.get("section"),
                    "highlights": highlights
                })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]
    
    def smart_search(self, query: str, limit: int = 10) -> List[Dict]:
        """Advanced search using BM25 with phrase matching and proximity scoring"""
        if not self.documents:
            return []
        
        self._rebuild_idf()
        query_lower = query.lower()
        query_words_stemmed = self._tokenize(query)
        query_words_original = self._tokenize(query, keep_original=True)
        
        if not query_words_stemmed and not query_words_original:
            return self.keyword_search(query, limit)
        
        # BM25 parameters
        k1 = 1.5
        b = 0.75
        
        total_len = sum(len(self._tokenize(text)) for text in self.documents.values())
        avg_doc_len = total_len / len(self.documents) if self.documents else 1
        
        results = []
        
        for chunk_id, text in self.documents.items():
            text_lower = text.lower()
            metadata = self.doc_metadata.get(chunk_id, {})
            doc_tf = self.tf_vectors.get(chunk_id, {})
            doc_words = self._tokenize(text)
            doc_len = len(doc_words)
            
            bm25_score = 0
            exact_match_bonus = 0
            original_word_bonus = 0
            phrase_bonus = 0
            proximity_bonus = 0
            
            # BM25 Score
            for word in query_words_stemmed:
                if word in doc_tf and word in self.idf:
                    tf = doc_tf[word] * doc_len
                    idf = self.idf[word]
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * (doc_len / avg_doc_len))
                    bm25_score += idf * (numerator / denominator)
            
            # Original word matches
            for word in query_words_original:
                if word in text_lower:
                    count = text_lower.count(word)
                    original_word_bonus += min(count * 0.3, 1.5)
            
            # Exact phrase match
            if len(query_lower) > 3 and query_lower in text_lower:
                exact_match_bonus = 3.0
            
            # Partial phrase matching
            if len(query_words_original) >= 2:
                for i in range(len(query_words_original) - 1):
                    bigram = f"{query_words_original[i]} {query_words_original[i+1]}"
                    if bigram in text_lower:
                        phrase_bonus += 0.8
                if len(query_words_original) >= 3:
                    for i in range(len(query_words_original) - 2):
                        trigram = f"{query_words_original[i]} {query_words_original[i+1]} {query_words_original[i+2]}"
                        if trigram in text_lower:
                            phrase_bonus += 1.2
            
            # Proximity scoring
            if len(query_words_original) >= 2:
                positions = {}
                for word in query_words_original:
                    pos = text_lower.find(word)
                    if pos != -1:
                        positions[word] = pos
                if len(positions) >= 2:
                    pos_values = list(positions.values())
                    min_dist = min(abs(pos_values[i] - pos_values[j]) 
                                   for i in range(len(pos_values)) 
                                   for j in range(i+1, len(pos_values)))
                    if min_dist < 100:
                        proximity_bonus = 1.0 - (min_dist / 100)
            
            total_score = (
                bm25_score * 0.35 +
                original_word_bonus * 0.25 +
                exact_match_bonus * 0.20 +
                phrase_bonus * 0.12 +
                proximity_bonus * 0.08
            )
            
            if total_score > 0.01:
                highlights = self._generate_highlights(text, query_words_original)
                results.append({
                    "chunk_id": chunk_id,
                    "text": text[:500] + "..." if len(text) > 500 else text,
                    "full_text": text,
                    "score": min(total_score / 3, 1.0),
                    "document_id": metadata.get("document_id", ""),
                    "document_name": metadata.get("filename", "Unknown"),
                    "page": metadata.get("page", 1),
                    "section": metadata.get("section"),
                    "highlights": highlights,
                    "match_type": "exact" if exact_match_bonus > 0 else "partial"
                })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]
    
    def hybrid_search(self, query: str, limit: int = 10, alpha: float = 0.7) -> List[Dict]:
        """Use smart_search as the primary method"""
        return self.smart_search(query, limit)
    
    def save_index(self, path: Path):
        """Save index to disk"""
        data = {
            "documents": self.documents,
            "doc_metadata": self.doc_metadata,
            "vocabulary": self.vocabulary,
            "tf_vectors": self.tf_vectors,
            "word_to_chunks": {k: list(v) for k, v in self.word_to_chunks.items()}
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
    
    def load_index(self, path: Path):
        """Load index from disk"""
        if not path.exists():
            return
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.documents = data.get("documents", {})
        self.doc_metadata = data.get("doc_metadata", {})
        self.vocabulary = data.get("vocabulary", {})
        self.tf_vectors = data.get("tf_vectors", {})
        self.word_to_chunks = defaultdict(set)
        for k, v in data.get("word_to_chunks", {}).items():
            self.word_to_chunks[k] = set(v)
