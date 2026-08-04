def main() -> None:
    from rag_with_langgraph.agent import run_agent

    print("Running RAG with LangGraph agent...")
    run_agent("What is LangGraph?")


__all__ = ["main"]
