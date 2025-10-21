from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Optional
import operator
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from chromadb.config import Settings
from datetime import datetime
import chromadb
import os

load_dotenv()

# ==================== STATE DEFINITION ====================
class AgentState(TypedDict):
    """State that flows through the graph"""
    messages: Annotated[List, operator.add]  # Conversation history
    user_input: str  # Current user input
    retrieved_context: Optional[str]  # Retrieved memories
    needs_memory: bool  # Whether to retrieve memories
    final_response: Optional[str]  # Response to user

# ==================== AZURE OPENAI CONFIGURATION ====================
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_GPT_4O_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_GPT_4O_DEPLOYMENT"),
    api_version=os.getenv("AZURE_OPENAI_GPT_4O_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_GPT_4O_API_KEY"),
    temperature=0.7,
)

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    api_version=os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY"),
)

class MemorySystem:

    def __init__(self, embeddings, collection_name = "conversation_memory"):
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.vectorstore = None
        self.chroma_client = chromadb.HttpClient(
            host="localhost",
            port=8000,
            settings=Settings(
                anonymized_telemetry=False
            )
        )

        self.load_memory()

    def load_memory(self):
        try:
            self.vectorstore = Chroma(
                client = self.chroma_client,
                collection_name=self.collection_name,
                embedding_function= self.embeddings,
            )

            if self.vectorstore._collection.count() == 0:
                docs = [Document(page_content="Memory initialized",
                        metadata={"timestamp": str(datetime.now()), "type": "system"})]
                self.vectorstore.add_documents(docs)
            else:
                count = self.vectorstore._collection.count()
        except Exception as e:
            print(f"Error connecting to ChromaDB: {e}")
            print("Make sure ChromaDB is running in Docker!")
            raise
    
    def add_interaction(self, user_input: str, assistant_response: str):
        interaction = f"User: {user_input}\nAssistant: {assistant_response}"
        metadata= {
            "timestamp": str(datetime.now()),
            "type": "conversation",
            "user_input": user_input[:100]
        }
        doc = Document(page_content=interaction, metadata=metadata)
        self.vectorstore.add_documents([doc])

    def retrieve_relevant_memories(self, query: str, k: int = 3) -> str:
        if not self.vectorstore:
            return "No previous memories found."

        docs = self.vectorstore.similarity_search(query, k, filter={"type":"conversation"})
        if not docs or docs[0].page_content == "Memory initialized":
            return "No relevant memories found."
        
        memories = "\n\n---\n\n".join([doc.page_content for doc in docs])
        return f"Relevant past conversations:\n{memories}"

memory_system = MemorySystem(embeddings)

def should_retrieve_memory(state: AgentState) -> AgentState:
    #TODO: figure out if there is a way to make this part better

    user_input = state["user_input"]

    memory_keywords = [
        "remember", "told you", "mentioned", "last time", 
        "previously", "before", "earlier", "said",
        "what did i", "do you recall", "my preference"
    ]
    
    needs_memory = any(keyword in user_input.lower() for keyword in memory_keywords)
    
    # Always retrieve some context for continuity
    state["needs_memory"] = True  # Can be True or needs_memory based on strategy
    
    return state

def retrieve_memory_node(state:AgentState) -> AgentState:
    if state["needs_memory"]:
        retrieved = memory_system.retrieve_relevant_memories(state["user_input"], k=3)
        state["retrieved_context"] = retrieved
    else:
        state["retrieved_context"] = "No memory retrieved"
    return state

def generate_response_node(state: AgentState) -> AgentState:
    """Generate response using LLM with context"""
    
    # Build context
    memory_context = state.get("retrieved_context", "")
    
    system_prompt = f"""You are a helpful personal assistant with memory.

RELEVANT MEMORIES:
{memory_context}

Instructions:
- Be conversational and remember past interactions
- Use the user's name if you know it
- Reference past conversations when relevant
- Be helpful, friendly, and personalized
"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{input}")
    ])
    
    chain = prompt | llm
    
    response = chain.invoke({
        "messages": state["messages"],
        "input": state["user_input"]
    })
    state["final_response"] = response.content
    
    # Save interaction to memory
    memory_system.add_interaction(state["user_input"], response.content)
    
    # Update messages
    state["messages"].append(HumanMessage(content=state["user_input"]))
    state["messages"].append(AIMessage(content=response.content))
    
    return state

def create_graph():

    workflow = StateGraph(AgentState)

    workflow.add_node("check_memory", should_retrieve_memory)
    workflow.add_node("retrieve_memory", retrieve_memory_node)
    workflow.add_node("generate_response", generate_response_node)


    workflow.set_entry_point("check_memory")
    workflow.add_edge("check_memory", "retrieve_memory")
    workflow.add_edge("retrieve_memory", "generate_response")
    workflow.add_edge("generate_response", END)
    
    return workflow.compile()



conversation_state = {
    "messages": [],
    "user_input": "",
    "retrieved_context": None,
    "needs_memory": False,
    "final_response": None,
}

app = create_graph()

conversation_count = 0

print("=" * 100)
print("🤖 PERSONAL ASSISTANT WITH MEMORY")
print("=" * 100)
print("Type 'exit' to quit, 'clear' to clear memory\n")

while True:
    user_input = input("You: ").strip()
    
    if user_input.lower() in ['exit', 'quit', 'q']:
        print("\n👋 Goodbye! Your memories have been saved.")
        memory_system.save_memory()
        break
    
    if user_input.lower() == 'clear':
        if os.path.exists(memory_system.memory_file):
            os.remove(memory_system.memory_file)
        memory_system.load_memory()
        conversation_state["messages"] = []
        print("🗑️  Memory cleared!\n")
        continue
    
    if not user_input:
        continue

    conversation_state["user_input"] = user_input

    result = app.invoke(conversation_state)
    
    # Update conversation state
    conversation_state = result
    
    # Display response
    print(f"\nAssistant: {result['final_response']}\n")
    print("-" * 100 + "\n")
    
    conversation_count += 1
