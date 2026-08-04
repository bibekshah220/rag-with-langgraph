import glob
import os
import sys
from pathlib import Path

# Ensure src directory is in sys.path
root_dir = Path(__file__).parent.resolve()
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import streamlit as st

# Load Streamlit Cloud secrets into environment variables
# (must happen BEFORE importing agent, which initializes ChatGroq at module level)
try:
    for key, value in st.secrets.items():
        if isinstance(value, str):
            os.environ.setdefault(key, value)
except Exception:
    pass  # Not on Streamlit Cloud, .env will be used instead

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from rag_with_langgraph.agent import (
    DOCS_DIR,
    INDEX_DIR,
    agent,
    build_faiss_index,
)

# Page Configuration
st.set_page_config(
    page_title="RAG with LangGraph",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4F46E5, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        color: #6B7280;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .stChatMessage {
        border-radius: 10px;
    }
</style>
""",
    unsafe_allow_html=True,
)


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


init_session_state()

# Sidebar Layout
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/brain--v1.png", width=64)
    st.markdown("### 🛠️ RAG & VectorStore Setup")

    # Document Stats
    os.makedirs(DOCS_DIR, exist_ok=True)
    doc_files = glob.glob(os.path.join(DOCS_DIR, "*.txt"))
    st.info(f"📁 **{len(doc_files)} files** in `{DOCS_DIR}/`")
    with st.expander("📄 Sample Documents List"):
        for f in doc_files:
            st.text(f"• {os.path.basename(f)}")

    st.divider()

    # Re-index Button
    if st.button("🔄 Rebuild FAISS Index", use_container_width=True, type="primary"):
        with st.spinner("Indexing documents with Gemini Embeddings..."):
            try:
                build_faiss_index(DOCS_DIR, INDEX_DIR)
                st.success("FAISS Index successfully rebuilt!")
            except Exception as e:
                st.error(f"Failed to rebuild index: {e}")

    st.divider()

    # Upload New Document
    st.markdown("### 📤 Upload New Document")
    uploaded_file = st.file_uploader("Upload a `.txt` file", type=["txt"])
    if uploaded_file is not None:
        save_path = os.path.join(DOCS_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Saved `{uploaded_file.name}` to `{DOCS_DIR}/`!")
        if st.button("⚡ Index New Document Now", use_container_width=True):
            with st.spinner("Updating index..."):
                build_faiss_index(DOCS_DIR, INDEX_DIR)
                st.success("Index updated with new document!")

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

# Main Chat View
st.markdown('<div class="main-title">🤖 RAG Assistant with LangGraph</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Powered by LangGraph, FAISS VectorStore, ChatGroq & Google Gemini Embeddings</div>',
    unsafe_allow_html=True,
)

# Render existing chat messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    elif msg["role"] == "assistant":
        with st.chat_message("assistant"):
            if "tool_calls" in msg and msg["tool_calls"]:
                for tc in msg["tool_calls"]:
                    with st.expander(f"⚙️ Tool Invoked: `{tc['name']}`"):
                        st.json(tc["args"])
            if "tool_outputs" in msg and msg["tool_outputs"]:
                for out in msg["tool_outputs"]:
                    with st.expander("📊 Tool Result"):
                        st.code(out, language="text")
            st.markdown(msg["content"])

# User Prompt Input
if prompt := st.chat_input("Ask about LangGraph, RAG, or arithmetic calculations..."):
    # Render user prompt
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.chat_history.append(HumanMessage(content=prompt))

    # Invoke Agent
    with st.chat_message("assistant"):
        with st.spinner("Agent is reasoning and executing tools..."):
            try:
                result = agent.invoke({"messages": st.session_state.chat_history, "llm_calls": 0})

                # Process agent output steps
                tool_calls_info = []
                tool_outputs_info = []
                final_answer = ""

                for msg in result["messages"]:
                    if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                        for tc in msg.tool_calls:
                            tool_calls_info.append(tc)
                            with st.expander(f"⚙️ Tool Invoked: `{tc['name']}`"):
                                st.json(tc["args"])

                    if isinstance(msg, ToolMessage):
                        tool_outputs_info.append(str(msg.content))
                        with st.expander("📊 Tool Result"):
                            st.code(
                                str(msg.content)[:400] + ("..." if len(str(msg.content)) > 400 else ""),
                                language="text",
                            )

                final_answer = str(result["messages"][-1].content)
                st.markdown(final_answer)

                # Store assistant response in session state
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": final_answer,
                        "tool_calls": tool_calls_info,
                        "tool_outputs": tool_outputs_info,
                    }
                )
                # Update full message chain history for conversational state
                st.session_state.chat_history = result["messages"]

            except Exception as e:
                st.error(f"Error running agent: {e}")
