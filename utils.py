import streamlit as st
from typing import List, Dict

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    
    if 'processed_documents' not in st.session_state:
        st.session_state.processed_documents = []
    
    if 'chat_handler' not in st.session_state:
        st.session_state.chat_handler = None

def display_chat_history():
    """Display chat history in Streamlit"""
    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
    else:
        with st.chat_message("assistant"):
            st.write("👋 Hello! Upload some documents and I'll help you find information from them.")

def format_document_stats(stats: dict) -> str:
    """Format document statistics for display"""
    if not stats:
        return "No statistics available"
    
    formatted = f"""
    **Document Statistics:**
    - Total chunks: {stats.get('total_chunks', 0)}
    - Total characters: {stats.get('total_characters', 0):,}
    - Average chunk size: {stats.get('average_chunk_size', 0)} characters
    - Unique sources: {stats.get('unique_sources', 0)}
    - File types: {', '.join(stats.get('file_types', []))}
    """
    return formatted

def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to specified length with ellipsis"""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

def validate_file_size(file, max_size_mb: int = 10) -> bool:
    """Validate uploaded file size"""
    if hasattr(file, 'size'):
        size_mb = file.size / (1024 * 1024)
        return size_mb <= max_size_mb
    return True

def get_file_type_icon(file_extension: str) -> str:
    """Get appropriate icon for file type"""
    icons = {
        '.pdf': '📄',
        '.txt': '📝',
        '.docx': '📃',
        '.doc': '📃'
    }
    return icons.get(file_extension.lower(), '📄')

def display_error_message(error: Exception, context: str = ""):
    """Display formatted error message"""
    error_msg = f"Error{f' in {context}' if context else ''}: {str(error)}"
    st.error(error_msg)
    
    # Log error details for debugging
    import logging
    logging.error(f"{context}: {str(error)}", exc_info=True)

def display_success_message(message: str, details: str = ""):
    """Display formatted success message"""
    st.success(message)
    if details:
        st.info(details)

def create_download_link(content: str, filename: str, link_text: str) -> str:
    """Create download link for text content"""
    import base64
    
    b64 = base64.b64encode(content.encode()).decode()
    href = f'<a href="data:text/plain;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

def format_chat_export(chat_history: List[Dict]) -> str:
    """Format chat history for export"""
    if not chat_history:
        return "No chat history available."
    
    formatted_chat = []
    formatted_chat.append("# Chat History Export")
    formatted_chat.append(f"Generated on: {st.timestamp}")
    formatted_chat.append("")
    
    for i, message in enumerate(chat_history, 1):
        role = message.get('role', 'unknown').title()
        content = message.get('content', '')
        formatted_chat.append(f"## Message {i} - {role}")
        formatted_chat.append(content)
        formatted_chat.append("")
    
    return "\n".join(formatted_chat)

def estimate_tokens(text: str) -> int:
    """Rough estimation of tokens in text"""
    # Simple approximation: 1 token ≈ 4 characters
    return len(text) // 4

def chunk_text_by_tokens(text: str, max_tokens: int = 1000) -> List[str]:
    """Split text into chunks based on token estimation"""
    max_chars = max_tokens * 4  # Rough conversion
    
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    words = text.split()
    current_chunk = []
    current_length = 0
    
    for word in words:
        word_length = len(word) + 1  # +1 for space
        
        if current_length + word_length > max_chars and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks
