import glob
import operator
import os
from typing import Annotated, Any, Dict, List, Literal, Sequence
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

# Fix OpenMP duplicate library issue on some OS environments
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load environment variables (.env)
load_dotenv()

DOCS_DIR = "sample_docs"
INDEX_DIR = "faiss_index"


def load_and_split(docs_dir: str = DOCS_DIR) -> List[Document]:
    """Load text files from docs_dir and split them into Document chunks."""
    chunks: List[Document] = []
    if not os.path.exists(docs_dir):
        return chunks
    for path in glob.glob(os.path.join(docs_dir, "*.txt")):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        source = os.path.basename(path)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        current = ""
        for para in paragraphs:
            if len(current) + len(para) + 2 <= 500:
                current = (current + "\n\n" + para).strip()
            else:
                if current:
                    chunks.append(Document(page_content=current, metadata={"source": source}))
                current = para
        if current:
            chunks.append(Document(page_content=current, metadata={"source": source}))
    return chunks


def build_faiss_index(docs_dir: str = DOCS_DIR, index_dir: str = INDEX_DIR) -> FAISS:
    """Build and save FAISS vector store index from sample_docs."""
    chunks = load_and_split(docs_dir)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_dir)
    print(f"Index created with {len(chunks)} chunks and saved to {index_dir}/")
    return vectorstore


def get_vectorstore(index_dir: str = INDEX_DIR) -> FAISS:
    """Get FAISS vectorstore from local disk or build if missing."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    if os.path.exists(index_dir):
        return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    else:
        print(f"{index_dir} not found. Building index...")
        return build_faiss_index(DOCS_DIR, index_dir)


# Define Tools
@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b


@tool
def divide(a: int, b: int) -> float:
    """Divides a by b."""
    return a / b


@tool
def search_docs(query: str) -> str:
    """Search the knowledge base for information about AI/ML concepts,
    LangGraph, RAG, embeddings, transformers, and related topics."""
    vectorstore = get_vectorstore(INDEX_DIR)
    docs = vectorstore.as_retriever(search_kwargs={"k": 3}).invoke(query)
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


tools = [add, multiply, divide, search_docs]
tools_by_name = {t.name: t for t in tools}

# Model Setup
model = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
model_with_tools = model.bind_tools(tools)


# Define Graph State
class MessagesState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
    llm_calls: int


# Graph Nodes & Routing
def llm_call(state: MessagesState) -> Dict[str, Any]:
    response = model_with_tools.invoke(
        [
            SystemMessage(
                content=(
                    "You are a helpful assistant that can perform arithmetic "
                    "and answer questions about AI/ML concepts. "
                    "Use search_docs for AI/ML questions, math tools for calculations."
                )
            )
        ]
        + list(state["messages"])
    )
    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def tool_node(state: MessagesState) -> Dict[str, Any]:
    results: List[ToolMessage] = []
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", []) or []
    for tool_call in tool_calls:
        t = tools_by_name[tool_call["name"]]
        observation = t.invoke(tool_call["args"])
        results.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
    return {"messages": results}


def should_continue(state: MessagesState) -> Literal["tool_node", "__end__"]:
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None)
    if tool_calls:
        return "tool_node"
    return END


# Build Graph
def create_agent():
    agent_builder = StateGraph(MessagesState)
    agent_builder.add_node("llm_call", llm_call)
    agent_builder.add_node("tool_node", tool_node)

    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    agent_builder.add_edge("tool_node", "llm_call")

    return agent_builder.compile()


agent = create_agent()


def run_agent(question: str) -> str:
    print(f"Q: {question}")
    result = agent.invoke({"messages": [HumanMessage(content=question)], "llm_calls": 0})
    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                print(f"  tool: {tc['name']}  args: {tc['args']}")
        if isinstance(msg, ToolMessage):
            print(f"  result: {msg.content[:120]}...")
    final_content = str(result["messages"][-1].content)
    print(f"A: {final_content}\n")
    return final_content


if __name__ == "__main__":
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "What is LangGraph?"
    run_agent(query)
