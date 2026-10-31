"""Streamlit user interface for the College Knowledge Assistant."""
import streamlit as st

from rag_pipeline import CollegeRAG

st.set_page_config(page_title="College Knowledge Assistant", page_icon="🎓", layout="wide")
st.title("🎓 College Knowledge Assistant")
st.caption("Ask questions about the sample college policies. Answers are grounded in indexed documents.")

with st.sidebar:
    st.header("About")
    st.write("This demo uses local embeddings, FAISS retrieval, and Ollama for local answer generation.")
    st.write("No paid API key is required.")
    st.warning("Sample policies are fictional and for software testing only.")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

@st.cache_resource
def load_rag():
    return CollegeRAG()

try:
    rag = load_rag()
except Exception as exc:
    st.error(str(exc))
    st.info("First run `python ingest.py` from the project folder.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Supporting sources"):
                for i, source in enumerate(message["sources"], start=1):
                    page = source["page"] if source["page"] is not None else "N/A (text file)"
                    st.markdown(
                        f"**{i}. {source['source']}** · page {page} · "
                        f"similarity `{source['similarity']}`"
                    )
                    st.caption(source["excerpt"])

question = st.chat_input("Ask about attendance, exams, library rules, scholarships…")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching college documents…"):
            result = rag.answer(question)
        st.markdown(result["answer"])
        if result["sources"]:
            with st.expander("Supporting sources", expanded=True):
                for i, source in enumerate(result["sources"], start=1):
                    page = source["page"] if source["page"] is not None else "N/A (text file)"
                    st.markdown(
                        f"**{i}. {source['source']}** · page {page} · "
                        f"similarity `{source['similarity']}`"
                    )
                    st.caption(source["excerpt"])
        if result["status"] in {"model_unavailable", "model_error"}:
            st.info("Retrieval worked, but answer generation needs attention.")
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })
