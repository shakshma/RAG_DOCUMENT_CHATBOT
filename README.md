# RAG_DOCUMENT_CHATBOT
"A Retrieval-Augmented Generation (RAG) chatbot built with LangChain, Sentence Transformers, FAISS, and Streamlit. This project enables question-answering over custom documents, leveraging local LLMs (via Ollama) for grounded and factual responses."
RAG Chatbot
This project implements a Retrieval-Augmented Generation (RAG) chatbot using LangChain, Sentence Transformers for embeddings, FAISS for vector storage, and Streamlit for the user interface. The chatbot can answer questions based on a set of provided documents.

Features
Document Processing: Ingests and processes text documents, converting them into searchable chunks.

Vector Database Creation: Generates embeddings for document chunks and stores them in a FAISS vector database for efficient similarity search.

Document Retrieval: Retrieves relevant document chunks based on user queries.

Response Generation: Uses a Large Language Model (LLM) (e.g., Mistral via Ollama, or OpenAI GPT models) to generate coherent answers grounded in the retrieved documents.

Streamlit UI: Provides an interactive web interface for users to ask questions and receive answers.

Project Structure
my_rag_chatbot/
├── data/                    # Directory for raw input documents (e.g., .txt, .pdf)
├── chunks/                  # Directory for processed document chunks (if saved)
├── vectordb/                # Directory to store the FAISS index files
│   └── faiss_index/         # Contains index.faiss and index.pkl
├── notebooks/
│   ├── 1_document_processing_and_embedding.ipynb # Notebook for data ingestion and FAISS index creation
│   └── 3_rag_pipeline_testing.ipynb            # Notebook for testing the RAG pipeline components
├── src/
│   ├── __init__.py          # Makes 'src' a Python package
│   ├── retriever.py         # Contains DocumentRetriever class
│   └── generator.py         # Contains ResponseGenerator class
├── app.py                   # The main Streamlit application file
├── requirements.txt         # List of Python dependencies
└── README.md                # This README file

Setup Instructions
Follow these steps to set up the project environment and install dependencies.

1. Clone the Repository (if applicable)
If this project is in a Git repository, clone it:

git clone <your-repository-url>
cd my_rag_chatbot

2. Create and Activate a Virtual Environment
It's highly recommended to use a virtual environment to manage project dependencies.

# Create a virtual environment named .venv
python -m venv .venv

# Activate the virtual environment
# On Windows (PowerShell):
.venv\Scripts\activate
# On macOS/Linux (Bash/Zsh):
source .venv/bin/activate

You should see (.venv) at the beginning of your terminal prompt, indicating the virtual environment is active.

3. Install Dependencies
With your virtual environment activated, install the required Python packages:

pip install -r requirements.txt

Note: If you encounter issues with langchain-community or SentenceTransformerEmbeddings, ensure ipykernel is also installed and registered:

pip install --upgrade ipykernel
python -m ipykernel install --user --name=.venv --display-name "Python (.venv)"

4. Install Ollama (for Local LLM - Optional but Recommended)
If you plan to use a local LLM like Mistral, download and install Ollama:

Download from: https://ollama.ai/download

After installation, pull the Mistral model:

ollama run mistral

This will download the model and start it. Keep this running in a separate terminal if you want your chatbot to use it.

How to Process Documents and Create FAISS Index
Before running the chatbot, you need to process your documents and create the FAISS vector database.

Place your documents (e.g., .txt, .pdf) into the data/ directory.

Open VS Code by selecting the my_rag_chatbot folder (File > Open Folder...).

Open the Jupyter notebook: notebooks/1_document_processing_and_embedding.ipynb.

Select the correct Jupyter Kernel: In the top-right corner of the notebook, click the kernel selector and choose Python (.venv).

Run all cells in the 1_document_processing_and_embedding.ipynb notebook. This will:

Load your documents.

Split them into chunks.

Generate embeddings.

Create and save the FAISS index to ./vectordb/faiss_index/.

How to Run the Chatbot Application
Once the FAISS index is created, you can run the Streamlit application.

Ensure your virtual environment is activated (as described in step 2 of Setup Instructions).

Ensure you are in the project's root directory (my_rag_chatbot). If you are in src or notebooks, use cd .. to go up.

Run the Streamlit app:

streamlit run app.py

Your browser should automatically open to http://localhost:8501, displaying the RAG Chatbot interface.

Troubleshooting Common Issues
ModuleNotFoundError: No module named 'src' or No module named 'langchain_community':

Kernel Mismatch: The most common cause. Ensure your VS Code Jupyter kernel is explicitly set to Python (.venv). Restart the kernel after changing.

Working Directory: Ensure you opened VS Code by selecting the my_rag_chatbot folder, and that streamlit run app.py is executed from the my_rag_chatbot directory.

Dependencies Not Installed: Make sure pip install -r requirements.txt ran successfully within your activated .venv.

TypeError: DocumentRetriever.__init__() got an unexpected keyword argument 'embeddings_model_name':

This means your src/retriever.py file is not updated or the kernel hasn't reloaded it. Double-check src/retriever.py matches the latest provided code and restart the kernel.

FAISS index not found at ./vectordb/faiss_index:

You must run notebooks/1_document_processing_and_embedding.ipynb successfully to create these files. Verify index.faiss and index.pkl exist in my_rag_chatbot/vectordb/faiss_index/.

LLM Issues (e.g., Ollama not found, connection errors):

If using Ollama, ensure the Ollama server is running and you have pulled the required model (e.g., ollama run mistral).

If using an API-based LLM (like OpenAI), ensure your API key is correctly set as an environment variable.
