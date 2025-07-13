# src/retriever.py

import os
from langchain_community.embeddings import SentenceTransformerEmbedding
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

class DocumentRetriever:
    """
    Manages the retrieval of relevant documents from a FAISS vector database.
    """
    def __init__(self, embeddings_model_name: str = "sentence-transformers/all-MiniLM-L6-v2", faiss_index_path: str = "./vectordb/faiss_index"):
        """
        Initializes the DocumentRetriever.

        Args:
            embeddings_model_name (str): The name of the SentenceTransformer model to use for embeddings.
                                         Defaults to "sentence-transformers/all-MiniLM-L6-v2".
            faiss_index_path (str): The path to the directory containing the FAISS index files.
                                    Defaults to "./vectordb/faiss_index" relative to the project root.
        """
        self.faiss_index_path = faiss_index_path
        self.embedding_model_name = embeddings_model_name # Use the passed argument
        self.embeddings = self._load_embedding_model()
        self.vector_db = self._load_vector_db()

    def _load_embedding_model(self):
        """Loads the SentenceTransformer embedding model."""
        print(f"Loading embedding model: {self.embedding_model_name} for retrieval...")
        return SentenceTransformerEmbeddings(model_name=self.embedding_model_name)

    def _load_vector_db(self):
        """Loads the FAISS vector database."""
        if not os.path.exists(self.faiss_index_path):
            raise FileNotFoundError(f"FAISS index not found at: {self.faiss_index_path}")
        print(f"Loading FAISS vector database from: {self.faiss_index_path}")
        # `allow_dangerous_deserialization=True` is needed for loading local pickle files
        # in recent LangChain versions due to security considerations.
        # Only set to True if you trust the source of the pickle file (i.e., you created it).
        return FAISS.load_local(
            self.faiss_index_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

    def get_relevant_documents(self, query: str, k: int = 5) -> list[Document]:
        """
        Retrieves the top-k most relevant documents for a given query.

        Args:
            query (str): The user's query.
            k (int): The number of top relevant documents to retrieve.

        Returns:
            list[Document]: A list of LangChain Document objects.
        """
        print(f"Retrieving {k} relevant documents for query: '{query}'")
        # You can use similarity_search_with_score if you want to see scores
        # docs_with_scores = self.vector_db.similarity_search_with_score(query, k=k)
        # for doc, score in docs_with_scores:
        #     print(f"Content: {doc.page_content[:100]}..., Score: {score}")
        return self.vector_db.similarity_search(query, k=k)

if __name__ == "__main__":
    # Example usage (for testing this module directly)
    # This assumes you have run the 1_document_processing_and_embedding.ipynb
    # and saved the FAISS index to ./vectordb/faiss_index
    try:
        # Initialize with default model name and FAISS path
        retriever = DocumentRetriever()
        query = "What is the capital of France?"
        relevant_docs = retriever.get_relevant_documents(query)

        print("\n--- Retrieved Document Contents ---")
        for i, doc in enumerate(relevant_docs):
            print(f"Document {i+1} (Source: {doc.metadata.get('source', 'N/A')} Page: {doc.metadata.get('page', 'N/A')}):")
            print(doc.page_content[:500] + "...") # Print first 500 characters
            print("-" * 50)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure you have run '1_document_processing_and_embedding.ipynb' to create the FAISS index.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
