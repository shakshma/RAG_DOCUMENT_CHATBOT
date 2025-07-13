# src/generator.py

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# Depending on how you're running Mistral (local, API, HuggingFace Hub),
# you might need different imports.
# For a generic HuggingFace model, you'd typically use HuggingFaceHub or a local pipeline.
# If you're using a local Ollama instance, it would be from langchain_community.llms import Ollama
# For simplicity and common setup, let's assume HuggingFaceHub or a placeholder for now.
# If you're running it locally via a framework like Ollama or Llama.cpp, adjust the import.
# For this example, we'll use a placeholder that mimics an LLM.
# If you intend to use HuggingFaceHub, you'll need to install 'huggingface_hub' and set 'HUGGINGFACEHUB_API_TOKEN'.

# Placeholder for actual LLM integration
# In a real scenario, you'd import the specific LLM class, e.g.:
# from langchain_community.llms import HuggingFaceHub # For HuggingFace models via API
# from langchain_openai import ChatOpenAI # If using OpenAI
# from langchain_community.llms import Ollama # If using Ollama locally

class ResponseGenerator:
    """
    Generates a response based on a given question and context using an LLM.
    """
    def __init__(self, llm_model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        """
        Initializes the ResponseGenerator with a specified LLM.

        Args:
            llm_model_name (str): The ID or name of the LLM to use.
                                  Defaults to "mistralai/Mistral-7B-Instruct-v0.2".
        """
        self.llm_model_name = llm_model_name
        print(f"Initializing ResponseGenerator with LLM: {self.llm_model_name}")

        # IMPORTANT: Replace this with your actual LLM initialization
        # based on how you plan to access the Mistral model.
        # Examples:
        # If using HuggingFaceHub (requires HUGGINGFACEHUB_API_TOKEN env var):
        # from langchain_community.llms import HuggingFaceHub
        # self.llm = HuggingFaceHub(repo_id=llm_model_name, model_kwargs={"temperature": 0.5, "max_length": 500})

        # If using a local Ollama instance (e.g., 'ollama run mistral'):
        from langchain_community.llms import Ollama
        self.llm = Ollama(model=llm_model_name, temperature=0.7) # Adjust host if not default

        # If using OpenAI (for testing, replace with your actual model if different):
        # from langchain_openai import ChatOpenAI
        # self.llm = ChatOpenAI(model_name=llm_model_name, temperature=0.7)

        # For a simple placeholder if no actual LLM integration is set up yet:
        # class DummyLLM:
        #     def invoke(self, prompt_value):
        #         return f"Dummy response for: {prompt_value.messages[-1].content}"
        # self.llm = DummyLLM()


    def generate_response(self, question: str, context: str) -> str:
        """
        Generates a response using the initialized LLM based on the provided context and question.

        Args:
            question (str): The user's question.
            context (str): The retrieved context documents.

        Returns:
            str: The generated response.
        """
        print(f"Generating response for question: '{question}' with context...")

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Use the following context to answer the question. If the answer is not in the context, state that you don't know."),
            ("human", "Context: {context}\nQuestion: {question}")
        ])

        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"context": context, "question": question})
        return response

if __name__ == "__main__":
    # Example usage for testing this module directly
    try:
        generator = ResponseGenerator() # Uses the default model
        dummy_context = "The quick brown fox jumps over the lazy dog. Dogs are mammals."
        dummy_question = "What kind of animal is a dog?"
        response = generator.generate_response(dummy_question, dummy_context)
        print(f"\nGenerated Response (direct test): {response}")

        generator_gpt = ResponseGenerator(llm_model_name="gpt-3.5-turbo")
        response_gpt = generator_gpt.generate_response(dummy_question, dummy_context)
        print(f"\nGenerated Response (GPT-3.5-turbo test): {response_gpt}")

    except Exception as e:
        print(f"An error occurred during direct generator testing: {e}")
        import traceback
        traceback.print_exc()
