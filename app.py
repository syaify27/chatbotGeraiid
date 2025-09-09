import streamlit as st
import os
from document_processor import DocumentProcessor
from vector_store import VectorStore
from chat_handler import ChatHandler
from utils import initialize_session_state, display_chat_history

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🤖 RAG Document Chatbot")
    st.markdown("Upload documents and ask questions about their content using Google Gemini AI")
    
    # Initialize session state
    initialize_session_state()
    
    # Check for Gemini API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("⚠️ GEMINI_API_KEY environment variable not found. Please set your Google Gemini API key.")
        st.stop()
    
    # Sidebar for document upload and management
    with st.sidebar:
        st.header("📁 Document Management")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=['pdf', 'txt', 'docx'],
            accept_multiple_files=True,
            help="Supported formats: PDF, TXT, DOCX"
        )
        
        # Process uploaded files
        if uploaded_files:
            if st.button("🔄 Process Documents", type="primary"):
                process_documents(uploaded_files)
        
        # Display processed documents
        if st.session_state.processed_documents:
            st.subheader("📚 Processed Documents")
            for doc in st.session_state.processed_documents:
                st.write(f"✅ {doc}")
        
        # Clear documents button
        if st.session_state.processed_documents:
            if st.button("🗑️ Clear All Documents", type="secondary"):
                clear_documents()
    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat Interface")
        
        # Display chat history
        display_chat_history()
        
        # Chat input
        if st.session_state.vector_store and st.session_state.processed_documents:
            user_question = st.chat_input("Ask a question about your documents...")
            
            if user_question:
                handle_user_question(user_question)
        else:
            st.info("📤 Please upload and process documents to start chatting.")
    
    with col2:
        st.header("ℹ️ Information")
        st.markdown("""
        **How to use:**
        1. Upload your documents (PDF, TXT, DOCX)
        2. Click "Process Documents"
        3. Ask questions about the content
        
        **Features:**
        - Document chunking and embedding
        - Vector similarity search
        - Context-aware responses
        - Chat history
        """)

def process_documents(uploaded_files):
    """Process uploaded documents and create vector store"""
    try:
        with st.spinner("Processing documents..."):
            # Initialize document processor
            doc_processor = DocumentProcessor()
            
            # Process each file
            all_chunks = []
            processed_files = []
            
            progress_bar = st.progress(0)
            total_files = len(uploaded_files)
            
            for i, uploaded_file in enumerate(uploaded_files):
                st.write(f"Processing: {uploaded_file.name}")
                
                # Save uploaded file temporarily
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                try:
                    # Process document
                    chunks = doc_processor.process_document(temp_path)
                    all_chunks.extend(chunks)
                    processed_files.append(uploaded_file.name)
                    
                    # Update progress
                    progress_bar.progress((i + 1) / total_files)
                    
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            
            if all_chunks:
                # Create vector store
                st.write("Creating vector embeddings...")
                vector_store = VectorStore()
                vector_store.create_vector_store(all_chunks)
                
                # Update session state
                st.session_state.vector_store = vector_store
                st.session_state.processed_documents = processed_files
                st.session_state.chat_handler = ChatHandler(vector_store)
                
                st.success(f"✅ Successfully processed {len(processed_files)} documents with {len(all_chunks)} chunks!")
            else:
                st.error("No content could be extracted from the uploaded documents.")
                
    except Exception as e:
        st.error(f"Error during document processing: {str(e)}")

def handle_user_question(question):
    """Handle user question and generate response"""
    try:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": question})
        
        with st.spinner("Generating response..."):
            # Get response from chat handler
            response = st.session_state.chat_handler.get_response(question)
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Rerun to update chat display
        st.rerun()
        
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")

def clear_documents():
    """Clear all processed documents and reset session state"""
    st.session_state.vector_store = None
    st.session_state.processed_documents = []
    st.session_state.chat_history = []
    st.session_state.chat_handler = None
    st.success("🗑️ All documents cleared!")
    st.rerun()

if __name__ == "__main__":
    main()
