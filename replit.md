# Overview

This is a RAG (Retrieval-Augmented Generation) Document Chatbot built with Streamlit that allows users to upload documents and ask questions about their content using Google Gemini AI. The application processes documents (PDF, TXT, DOCX), creates vector embeddings for semantic search, and provides intelligent responses based on the document content.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Streamlit Framework**: Web-based user interface with sidebar for document management and main area for chat interactions
- **Session State Management**: Persistent storage of chat history, vector store, processed documents, and chat handler across user sessions
- **File Upload Interface**: Multi-file upload support for PDF, TXT, and DOCX formats with validation

## Backend Architecture
- **Modular Design**: Separated into distinct components for document processing, vector storage, and chat handling
- **Document Processing Pipeline**: 
  - Loads documents using LangChain loaders (PyPDFLoader, TextLoader, Docx2txtLoader)
  - Splits text into chunks using RecursiveCharacterTextSplitter with configurable chunk size (1000) and overlap (200)
- **Vector Storage System**:
  - FAISS-based vector database for efficient similarity search
  - Sentence Transformers (all-MiniLM-L6-v2) for embedding generation
  - Normalized embeddings for improved search accuracy
- **Chat Handler**: 
  - Google Gemini AI integration for response generation
  - RAG implementation that retrieves relevant context before generating responses
  - System prompt engineering for context-aware responses

## Data Processing Flow
1. **Document Ingestion**: Upload and validation of supported file formats
2. **Text Extraction**: Format-specific loaders extract raw text content
3. **Text Chunking**: Recursive splitting with overlap to maintain context
4. **Embedding Generation**: Sentence transformer creates vector representations
5. **Vector Indexing**: FAISS index stores embeddings for fast retrieval
6. **Query Processing**: User questions are embedded and matched against document vectors
7. **Response Generation**: Retrieved context is provided to Gemini for answer generation

## Design Patterns
- **Factory Pattern**: Different document loaders based on file extensions
- **Strategy Pattern**: Configurable text splitting and embedding strategies
- **Repository Pattern**: Vector store abstraction for document retrieval
- **Command Pattern**: Chat handler processes user queries with context injection

# External Dependencies

## AI/ML Services
- **Google Gemini AI**: Primary language model for response generation (gemini-2.5-flash)
- **Sentence Transformers**: Local embedding model for vector generation (all-MiniLM-L6-v2)

## Document Processing
- **LangChain**: Document loading and text splitting utilities
- **PyPDF**: PDF document processing
- **python-docx**: DOCX file handling

## Vector Database
- **FAISS**: Facebook AI Similarity Search for vector storage and retrieval
- **NumPy**: Numerical operations for embedding manipulation

## Web Framework
- **Streamlit**: Complete web application framework for UI and session management

## Configuration
- **Environment Variables**: GEMINI_API_KEY required for AI service authentication
- **File System**: Local temporary storage for uploaded documents during processing

## Performance Considerations
- Batch processing for embedding generation (batch_size=32)
- Normalized embeddings for improved similarity search
- Configurable context window (max_context_tokens=3000) for response generation
- In-memory vector storage with FAISS for fast retrieval