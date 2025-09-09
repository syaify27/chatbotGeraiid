import os
import logging
from typing import List, Optional
from google import genai
from google.genai import types
from vector_store import VectorStore

class ChatHandler:
    """Handles chat interactions and response generation"""
    
    def __init__(self, vector_store: VectorStore, model_name: str = "gemini-2.5-flash"):
        """
        Initialize chat handler
        
        Args:
            vector_store: VectorStore instance for retrieving context
            model_name: Gemini model name
        """
        self.vector_store = vector_store
        self.model_name = model_name
        
        # Initialize Gemini client
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not found")
        
        self.client = genai.Client(api_key=api_key)
        
        # System prompt for RAG
        self.system_prompt = """You are a helpful AI assistant that answers questions based on the provided document context. 

Instructions:
1. Use ONLY the information provided in the context to answer questions
2. If the context doesn't contain enough information to answer the question, say so clearly
3. Be accurate and cite specific parts of the context when possible
4. If asked about something not in the context, explain that you can only answer based on the uploaded documents
5. Provide clear, concise, and helpful responses
6. When referencing information, mention the source document when available

Remember: You can only answer questions based on the uploaded documents and their content."""
        
        logging.info(f"ChatHandler initialized with model: {model_name}")
    
    def get_response(self, question: str, max_context_tokens: int = 3000) -> str:
        """
        Generate response to user question using RAG
        
        Args:
            question: User's question
            max_context_tokens: Maximum tokens for context
            
        Returns:
            Generated response
        """
        try:
            # Get relevant context from vector store
            context = self.vector_store.get_relevant_context(question, max_context_tokens)
            
            if not context:
                return "I don't have any relevant information in the uploaded documents to answer your question. Please make sure you've uploaded documents that contain information about your topic."
            
            # Create prompt with context
            prompt = self._create_prompt(question, context)
            
            # Generate response using Gemini
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Content(
                        role="user", 
                        parts=[types.Part(text=prompt)]
                    )
                ],
                config=types.GenerateContentConfig(
                    system_instruction=self.system_prompt,
                    temperature=0.3,
                    max_output_tokens=1000
                )
            )
            
            if response.text:
                return response.text.strip()
            else:
                return "I apologize, but I couldn't generate a response. Please try rephrasing your question."
                
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return f"An error occurred while generating the response: {str(e)}"
    
    def _create_prompt(self, question: str, context: str) -> str:
        """
        Create prompt for Gemini with question and context
        
        Args:
            question: User's question
            context: Relevant context from documents
            
        Returns:
            Formatted prompt
        """
        prompt = f"""Context from uploaded documents:
{context}

Question: {question}

Please answer the question based on the provided context. If the context doesn't contain sufficient information to answer the question, please state that clearly."""
        
        return prompt
    
    def get_conversation_summary(self, chat_history: List[dict], max_length: int = 500) -> str:
        """
        Generate a summary of the conversation
        
        Args:
            chat_history: List of chat messages
            max_length: Maximum length of summary
            
        Returns:
            Conversation summary
        """
        try:
            if not chat_history:
                return "No conversation history available."
            
            # Format chat history
            conversation = []
            for message in chat_history[-10:]:  # Last 10 messages
                role = message.get('role', 'unknown')
                content = message.get('content', '')
                conversation.append(f"{role.capitalize()}: {content}")
            
            conversation_text = "\n".join(conversation)
            
            prompt = f"Please provide a brief summary of this conversation:\n\n{conversation_text}"
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=200
                )
            )
            
            return response.text if response.text else "Could not generate summary."
            
        except Exception as e:
            logging.error(f"Error generating conversation summary: {str(e)}")
            return "Error generating summary."
    
    def suggest_questions(self, num_suggestions: int = 3) -> List[str]:
        """
        Suggest relevant questions based on the uploaded documents
        
        Args:
            num_suggestions: Number of questions to suggest
            
        Returns:
            List of suggested questions
        """
        try:
            if not self.vector_store.documents:
                return []
            
            # Get sample content from documents
            sample_content = []
            for doc in self.vector_store.documents[:5]:  # First 5 documents
                content = doc.page_content[:300]  # First 300 chars
                source = doc.metadata.get('source', 'Unknown')
                sample_content.append(f"From {source}: {content}")
            
            content_text = "\n\n".join(sample_content)
            
            prompt = f"""Based on the following document content, suggest {num_suggestions} relevant questions that a user might want to ask:

{content_text}

Please provide {num_suggestions} specific, relevant questions that can be answered based on this content. Format as a numbered list."""
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.5,
                    max_output_tokens=300
                )
            )
            
            if response.text:
                # Parse response into list
                suggestions = []
                lines = response.text.strip().split('\n')
                for line in lines:
                    # Remove numbering and clean up
                    clean_line = line.strip()
                    if clean_line and (clean_line[0].isdigit() or clean_line.startswith('-')):
                        # Remove numbering
                        clean_question = clean_line.split('.', 1)[-1].strip()
                        if clean_question:
                            suggestions.append(clean_question)
                
                return suggestions[:num_suggestions]
            
            return []
            
        except Exception as e:
            logging.error(f"Error suggesting questions: {str(e)}")
            return []
