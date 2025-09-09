import os
import logging
from typing import List, Optional
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

class DocumentProcessor:
    """Handles document loading and processing for RAG system"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize document processor
        
        Args:
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
    def load_document(self, file_path: str) -> List[Document]:
        """
        Load document based on file extension
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of Document objects
        """
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_extension == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_extension == '.docx':
                loader = Docx2txtLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata['source'] = os.path.basename(file_path)
                doc.metadata['file_type'] = file_extension
            
            return documents
            
        except Exception as e:
            logging.error(f"Error loading document {file_path}: {str(e)}")
            raise
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks
        
        Args:
            documents: List of Document objects to split
            
        Returns:
            List of chunked Document objects
        """
        try:
            chunks = self.text_splitter.split_documents(documents)
            
            # Add chunk metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata['chunk_id'] = i
                chunk.metadata['chunk_size'] = len(chunk.page_content)
            
            return chunks
            
        except Exception as e:
            logging.error(f"Error splitting documents: {str(e)}")
            raise
    
    def process_document(self, file_path: str) -> List[Document]:
        """
        Complete document processing pipeline
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of processed and chunked Document objects
        """
        try:
            # Load document
            documents = self.load_document(file_path)
            
            if not documents:
                raise ValueError("No content could be extracted from the document")
            
            # Split into chunks
            chunks = self.split_documents(documents)
            
            # Filter out very small chunks
            filtered_chunks = [
                chunk for chunk in chunks 
                if len(chunk.page_content.strip()) > 50
            ]
            
            logging.info(f"Processed {file_path}: {len(documents)} documents -> {len(filtered_chunks)} chunks")
            
            return filtered_chunks
            
        except Exception as e:
            logging.error(f"Error processing document {file_path}: {str(e)}")
            raise
    
    def get_document_stats(self, chunks: List[Document]) -> dict:
        """
        Get statistics about processed documents
        
        Args:
            chunks: List of document chunks
            
        Returns:
            Dictionary with document statistics
        """
        if not chunks:
            return {}
        
        total_chars = sum(len(chunk.page_content) for chunk in chunks)
        avg_chunk_size = total_chars / len(chunks)
        
        sources = set(chunk.metadata.get('source', 'Unknown') for chunk in chunks)
        file_types = set(chunk.metadata.get('file_type', 'Unknown') for chunk in chunks)
        
        return {
            'total_chunks': len(chunks),
            'total_characters': total_chars,
            'average_chunk_size': round(avg_chunk_size, 2),
            'unique_sources': len(sources),
            'sources': list(sources),
            'file_types': list(file_types)
        }
