import os
import logging
from typing import List, Optional, Tuple, Any
import re
from google import genai
from google.genai import types

class VectorStore:
    """Simplified vector store using Google Gemini for embeddings and text similarity"""
    
    def __init__(self):
        """Initialize vector store with Gemini client"""
        # Initialize Gemini client for embeddings
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not found")
        
        self.client = genai.Client(api_key=api_key)
        self.documents: List = []
        
        logging.info("Initialized VectorStore with Gemini embeddings")
    
    def create_vector_store(self, documents: List) -> None:
        """
        Create simple vector store from documents
        
        Args:
            documents: List of Document objects
        """
        try:
            if not documents:
                raise ValueError("No documents provided")
            
            # Store documents directly for simple text-based search
            self.documents = documents
            
            logging.info(f"Vector store created with {len(documents)} documents")
            
        except Exception as e:
            logging.error(f"Error creating vector store: {str(e)}")
            raise
    
    def similarity_search(self, query: str, k: int = 5) -> List[Tuple[Any, float]]:
        """
        Perform simple text similarity search
        
        Args:
            query: Query string
            k: Number of similar documents to return
            
        Returns:
            List of tuples (document, similarity_score)
        """
        try:
            if not self.documents:
                raise ValueError("Vector store not initialized")
            
            # Simple keyword-based similarity search
            query_words = set(re.findall(r'\w+', query.lower()))
            results = []
            
            for doc in self.documents:
                if hasattr(doc, 'page_content'):
                    content = doc.page_content.lower()
                    doc_words = set(re.findall(r'\w+', content))
                    
                    # Calculate simple similarity score based on word overlap
                    if query_words:
                        similarity = len(query_words.intersection(doc_words)) / len(query_words)
                        
                        # Boost score if query appears as substring
                        if query.lower() in content:
                            similarity += 0.5
                        
                        results.append((doc, similarity))
            
            # Sort by similarity score and return top k
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:k]
            
        except Exception as e:
            logging.error(f"Error in similarity search: {str(e)}")
            raise
    
    def get_relevant_context(self, query: str, max_tokens: int = 3000) -> str:
        """
        Get relevant context for a query with token limit
        
        Args:
            query: Query string
            max_tokens: Maximum number of tokens in context
            
        Returns:
            Concatenated relevant context
        """
        try:
            # Get similar documents
            similar_docs = self.similarity_search(query, k=10)
            
            # Build context within token limit
            context_parts = []
            current_length = 0
            
            for doc, score in similar_docs:
                if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
                    content = doc.page_content.strip()
                    source = doc.metadata.get('source', 'Unknown')
                    
                    # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
                    estimated_tokens = len(content) // 4
                    
                    if current_length + estimated_tokens > max_tokens:
                        break
                    
                    context_parts.append(f"[Source: {source}]\n{content}")
                    current_length += estimated_tokens
            
            return "\n\n---\n\n".join(context_parts)
            
        except Exception as e:
            logging.error(f"Error getting relevant context: {str(e)}")
            return ""
    
    def get_stats(self) -> dict:
        """
        Get vector store statistics
        
        Returns:
            Dictionary with statistics
        """
        if not self.documents:
            return {"status": "not_initialized"}
        
        return {
            "status": "initialized",
            "total_documents": len(self.documents),
            "search_method": "keyword_similarity"
        }
    
    def clear(self) -> None:
        """Clear the vector store"""
        self.documents = []
        logging.info("Vector store cleared")
