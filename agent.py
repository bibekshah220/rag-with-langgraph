import sys
from pathlib import Path

# Add src to sys.path
src_path = str(Path(__file__).parent.resolve() / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from rag_with_langgraph.agent import (
    DOCS_DIR,
    INDEX_DIR,
    MessagesState,
    add,
    agent,
    build_faiss_index,
    create_agent,
    divide,
    get_vectorstore,
    load_and_split,
    multiply,
    run_agent,
    search_docs,
)

__all__ = [
    "agent",
    "create_agent",
    "run_agent",
    "build_faiss_index",
    "get_vectorstore",
    "load_and_split",
    "add",
    "multiply",
    "divide",
    "search_docs",
    "MessagesState",
    "DOCS_DIR",
    "INDEX_DIR",
]

if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "What is LangGraph?"
    run_agent(query)
