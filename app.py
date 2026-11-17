import os
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.types import Command

from src.graph import create_review_graph
from src.ingestion import HybridCodeRetriever, clone_repository, parse_codebase_ast


def _load_server_secrets() -> None:
    """Load Streamlit Cloud secrets without rendering them in the browser."""
    for name in (
        "GROQ_API_KEY",
        "LANGCHAIN_TRACING_V2",
        "LANGCHAIN_PROJECT",
        "LANGCHAIN_API_KEY",
    ):
        if os.getenv(name):
            continue
        try:
            value = st.secrets.get(name)
        except FileNotFoundError:
            value = None
        if value:
            os.environ[name] = str(value)


load_dotenv()
_load_server_secrets()
st.set_page_config(page_title="CodeGraph-Sentinel", layout="wide")
st.title("CodeGraph-Sentinel: Multi-Agent Codebase Reviewer")
st.caption("AST parsing | RRF hybrid search | Groq inference | LangGraph HITL")

for key, default in {"graph": None, "repo_stats": None, "messages": [], "pending_interrupt": None,
                     "last_metrics": None}.items():
    st.session_state.setdefault(key, default)
st.session_state.setdefault("thread_id", str(uuid.uuid4()))

with st.sidebar:
    st.header("1. API & Repository Setup")
    if os.getenv("GROQ_API_KEY"):
        st.success("Groq API configured securely.")
    else:
        st.error("Configure GROQ_API_KEY in Streamlit Secrets.")
    repo_url = st.text_input("Public GitHub Repository URL", value="https://github.com/pallets/click")
    if st.button("Clone & Index Codebase", type="primary", use_container_width=True):
        if not os.getenv("GROQ_API_KEY"):
            st.error("Please enter your GROQ_API_KEY first.")
        else:
            with st.spinner("Cloning, parsing AST blocks, and building the local index..."):
                try:
                    docs = parse_codebase_ast(clone_repository(repo_url))
                    retriever = HybridCodeRetriever(docs)
                    st.session_state.graph = create_review_graph(retriever)
                    st.session_state.repo_stats = {"url": repo_url, "ast_chunks": len(docs)}
                    st.session_state.messages = []
                    st.session_state.pending_interrupt = None
                    st.success(f"Indexed {len(docs)} AST chunks.")
                except Exception as exc:
                    st.error(f"Ingestion failed: {exc}")
    if st.session_state.repo_stats:
        st.metric("Indexed AST Chunks", st.session_state.repo_stats["ast_chunks"])
    if st.session_state.last_metrics:
        st.divider()
        st.header("2. Node Latency Telemetry")
        timings = st.session_state.last_metrics["timings"]
        st.metric("End-to-End Latency", f"{sum(timings.values()):.1f} ms")
        for name, value in timings.items():
            st.text(f"• {name}: {value} ms")
    if st.session_state.graph:
        st.divider()
        with st.expander("View LangGraph Topology"):
            st.code(st.session_state.graph.get_graph().draw_mermaid(), language="mermaid")

if not st.session_state.graph:
    st.info("Configure the Groq secret and repository URL, then click **Clone & Index Codebase**.")
    st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.pending_interrupt:
    st.warning("Human-in-the-Loop Gate: review the proposed refactoring diff.")
    st.markdown(st.session_state.pending_interrupt["proposed_diff"])
    approve, reject = st.columns(2)
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    if approve.button("Approve & Finalize Patch", type="primary", use_container_width=True):
        result = st.session_state.graph.invoke(Command(resume={"approved": True}), config)
        st.session_state.messages.append({"role": "assistant", "content": result["final_response"]})
        st.session_state.pending_interrupt = None
        st.rerun()
    if reject.button("Reject Patch", use_container_width=True):
        result = st.session_state.graph.invoke(Command(resume={"approved": False}), config)
        st.session_state.messages.append({"role": "assistant", "content": result["final_response"]})
        st.session_state.pending_interrupt = None
        st.rerun()

if prompt := st.chat_input("Ask about architecture, audit for bugs, or request a refactor..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    with st.chat_message("assistant"):
        with st.status("Executing Multi-Agent LangGraph...", expanded=True) as status:
            for event in st.session_state.graph.stream({"question": prompt}, config, stream_mode="updates"):
                for node_name in event:
                    st.write(f"Executed node: **{node_name}**")
            status.update(label="Execution step complete", state="complete", expanded=False)
        snapshot = st.session_state.graph.get_state(config)
        values = snapshot.values
        st.session_state.last_metrics = {"timings": values.get("node_timings", {})}
        if snapshot.next:
            st.session_state.pending_interrupt = {
                "proposed_diff": values.get("final_response", "No proposal generated.")
            }
            st.rerun()
        answer = values.get("final_response", "No response generated.")
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()
