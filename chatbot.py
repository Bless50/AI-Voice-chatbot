from llama_index.llms.groq import Groq
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent import ReActAgent
from llama_index.core import PromptTemplate
load_dotenv()

#Initializing the embedding model
model_name = "models/embedding-001"
Settings.embed_model = GeminiEmbedding(
    model_name=model_name
)

# Initialize the language model (Groq)
Settings.llm = Groq(temperature=0.0, model="llama-3.1-70b-versatile")

documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

knowledge_tool = QueryEngineTool.from_defaults(
    query_engine,
    name="PDFBIO.EL28.pdf",
    description="A RAG engine that contains health related information",
)

# Define a simplified system prompt
simplified_system_prompt = """
You are a health assistant designed to provide accurate and helpful information on health-related topics.

- Use the knowledge tool for detailed answers based on the provided documents.
- If the knowledge tool is not necessary, respond based on your own knowledge.
- Always prioritize accuracy and evidence-based information.
- Maintain a friendly and supportive tone.
- Keep your responses brief and to the point; aim for clarity and conciseness.
- If you do not know the answer, politely inform the user without elaborating unnecessarily.
"""

# Create a PromptTemplate instance
react_system_prompt = PromptTemplate(simplified_system_prompt)

agent = ReActAgent.from_tools(
    [knowledge_tool],
    verbose=True,
    
)
# Update the agent's prompts with your custom prompt
agent.update_prompts({"agent_worker:system_prompt": react_system_prompt})

#function to generate the response from the model
def get_response(user_input):
    """Get response from chat engine."""
    try:
        response = agent.chat(user_input)
        return str(response)
    except Exception as e:
        print(f"Error in get_response: {e}")
        return "I'm sorry, I encountered an error while processing your request."