from pymongo import MongoClient, TEXT
from typing import List, Dict, Any
import re

class MongoDBSearchService:

    def __init__(self, mongodb_url: str, database_name: str):
        self.client = MongoClient(mongodb_url)
        self.db = self.client[database_name]
        self.documents_collection = self.db["documents"]
        self.chunks_collection = self.db["chunks"]

    def search_chunks(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        try:
            text_results = list(self.chunks_collection.find(
                {"$text": {"$search": query}},
                {"_id": 0, "score": {"$meta": "textScore"}}
            ).sort([("score", {"$meta": "textScore"})]).limit(limit))

            if text_results:

                formatted_results = []
                for result in text_results:
                    formatted_results.append({
                        "chunk_id": result["chunk_id"],
                        "text": result["text"],
                        "document_id": result["doc_id"],
                        "page": result["page"],
                        "score": result.get("score", 0.5),
                        "document_name": self._get_document_title(result["doc_id"]),
                        "full_text": result["text"]
                    })
                return formatted_results
        except Exception as e:
            print(f"Text search failed: {e}")

        regex_results = list(self.chunks_collection.find(
            {"text": {"$regex": query, "$options": "i"}},
            {"_id": 0}
        ).limit(limit))

      
        formatted_results = []
        for result in regex_results:
            formatted_results.append({
                "chunk_id": result["chunk_id"],
                "text": result["text"],
                "document_id": result["doc_id"],
                "page": result["page"],
                "score": 0.8,  
                "document_name": self._get_document_title(result["doc_id"]),
                "full_text": result["text"]
            })

        return formatted_results

    def _get_document_title(self, doc_id: str) -> str:
        """Get document title by doc_id"""
        try:
            doc = self.documents_collection.find_one(
                {"doc_id": doc_id},
                {"_id": 0, "title": 1}
            )
            return doc.get("title", "Unknown Document") if doc else "Unknown Document"
        except:
            return "Unknown Document"

    def get_document_info(self, doc_id: str) -> Dict[str, Any]:
        """Get document information"""
        try:
            return self.documents_collection.find_one(
                {"doc_id": doc_id},
                {"_id": 0}
            ) or {}
        except:
            return {}

    def get_chunk_by_id(self, chunk_id: str) -> Dict[str, Any]:
        """Get a specific chunk by ID"""
        try:
            return self.chunks_collection.find_one(
                {"chunk_id": chunk_id},
                {"_id": 0}
            ) or {}
        except:
            return {}

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents"""
        try:
            return list(self.documents_collection.find({}, {"_id": 0}))
        except:
            return []

    def get_chunks_for_document(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document"""
        try:
            return list(self.chunks_collection.find(
                {"doc_id": doc_id},
                {"_id": 0}
            ).sort("page", 1))
        except:
            return []

    def search_documents(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for documents by title, author, or tags"""
        try:
            results = list(self.documents_collection.find({
                "$or": [
                    {"title": {"$regex": query, "$options": "i"}},
                    {"author": {"$regex": query, "$options": "i"}},
                    {"tags": {"$regex": query, "$options": "i"}}
                ]
            }, {"_id": 0}).limit(limit))

            return results
        except Exception as e:
            print(f"Document search failed: {e}")
            return []
