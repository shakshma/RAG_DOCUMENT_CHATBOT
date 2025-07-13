# app.py

import streamlit as st
import os
import sys

# --- Environment Setup for Streamlit ---
# Ensure the project root is in sys.path for module imports
# This is crucial for Streamlit to find modules in 'src'
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(current_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    # print(f"Added project root to sys.path: {project_root}") # Streamlit doesn't show print to console by default

# Import DocumentRetriever and ResponseGenerator from your src directory
# These imports should now work because the project root is in sys.path
try:
    from src.retriever import DocumentRetriever
    from src.generator import ResponseGenerator
    from langchain_core.documents import Document # For type hinting
except ImportError as e:
    st.error(f"Failed to import core modules from 'src'. Please check your project structure and Python environment. Error: {e}")
    st.stop() # Stop the app if core modules can't be imported

# --- Streamlit App Configuration ---
st.set_page_config(page_title="RAG Chatbot", page_icon="💬", layout="centered")

st.title("💬 RAG Chatbot")
st.markdown("Ask me anything about your documents!")

# --- Caching expensive operations ---
# Use Streamlit's caching to load models and databases only once
@st.cache_resource
def load_retriever():
    """
    Loads and returns the DocumentRetriever instance.
    """
    try:
        # Initialize DocumentRetriever with default values or specific paths if needed
        # The embeddings_model_name is now handled within DocumentRetriever's __init__
        retriever = DocumentRetriever(
            faiss_index_path="./vectordb/faiss_index"
        )
        return retriever
    except FileNotFoundError as e:
        st.error(f"Error loading document retriever: {e}")
        st.info("Please ensure you have run '1_document_processing_and_embedding.ipynb' to create the FAISS index.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the document retriever: {e}")
        st.stop()

@st.cache_resource
def load_generator():
    """
    Loads and returns the ResponseGenerator instance.
    """
    try:
        # Initialize ResponseGenerator with the default Mistral model
        # or specify 'gpt-3.5-turbo' if you have OpenAI configured
        generator = ResponseGenerator(llm_model_name="mistralai/Mistral-7B-Instruct-v0.2")
        return generator
    except Exception as e:
        st.error(f"Error loading response generator: {e}")
        st.info("Please ensure your LLM setup (e.g., Ollama running 'mistral', or OpenAI API key) is correct.")
        st.stop()

# Load the retriever and generator instances
retriever = load_retriever()
generator = load_generator()

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What would you like to know?"):
    # Display user message in chat message container
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Thinking..."):
        # 1. Retrieve relevant documents
        try:
            relevant_docs = retriever.get_relevant_documents(prompt, k=5)
            context = " ".join([doc.page_content for doc in relevant_docs])
            st.session_state.messages.append({"role": "assistant", "content": f"**Retrieved Context:**\n```\n{context[:500]}...\n```"})
        except Exception as e:
            st.error(f"Error during document retrieval: {e}")
            st.stop()

        # 2. Generate response using the LLM
        try:
            response = generator.generate_response(prompt, context)
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
        except Exception as e:
            st.error(f"Error during response generation: {e}")
            st.stop()

# --- Sidebar for additional info/debug ---
with st.sidebar:
    st.header("App Status & Info")
    st.write(f"Project Root: {project_root}")
    st.write(f"Python Executable: {sys.executable}")
    st.write(f"FAISS Index Path: {retriever.faiss_index_path if 'retriever' in locals() else 'Not loaded'}")
    st.write(f"Embedding Model: {retriever.embedding_model_name if 'retriever' in locals() else 'Not loaded'}")
    st.write(f"LLM Model: {generator.llm_model_name if 'generator' in locals() else 'Not loaded'}")

    st.markdown("---")
    st.markdown("### Debugging Tips:")
    st.markdown("- Ensure Ollama is running (`ollama run mistral`) if using local LLM.")
    st.markdown("- Verify `vectordb/faiss_index` exists and contains `index.faiss` and `index.pkl`.")
    st.markdown("- Check your terminal for any warnings or errors after launching Streamlit.")

