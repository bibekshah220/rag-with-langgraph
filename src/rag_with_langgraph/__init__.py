from rag_with_langgraph.agent import agent, create_agent, run_agent


def main() -> None:
    print("Running RAG with LangGraph agent...")
    run_agent("What is LangGraph?")


__all__ = ["agent", "run_agent", "create_agent", "main"]
