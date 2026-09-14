"""
Stage 15 - Streamlit UI - SRC version with lazy load
Run: streamlit run src/app.py --server.fileWatcherType none
"""
import streamlit as st
from config import RETRIEVAL_TOP_K, SIMILARITY_THRESHOLD

st.set_page_config(page_title="Generative QA - RAG", layout="wide")
st.title("Question Answering System using NLP and Generative QA")
st.caption(f"RAG + FLAN-T5-base | top_k={RETRIEVAL_TOP_K} | threshold={SIMILARITY_THRESHOLD} | CPU-only offline")

with st.sidebar:
    st.metric("Threshold", SIMILARITY_THRESHOLD)
    st.metric("Top-K", RETRIEVAL_TOP_K)

examples = [
    "Explain tokenization in simple words.",
    "What is retrieval augmented generation?",
    "What is hallucination in language models?",
    "What is the best recipe for chocolate cake?",
]

q = st.text_input("Ask a question:", placeholder="Explain tokenization...")
use_example = st.selectbox("Try example:", [""] + examples)
if use_example:
    q = use_example

def load_pipeline():
    from qa_pipeline import answer_question
    return answer_question

if st.button("Get Answer") and q:
    with st.spinner("Loading... first ~15s"):
        try:
            fn = load_pipeline()
            result = fn(q)
        except Exception as e:
            st.error(f"Load failed: {e}")
            st.stop()

    st.subheader("Answer")
    st.success(result["answer"]) if result["grounded"] else st.warning(result["answer"])
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Top Score", f"{result['top_score']:.3f}")
    c2.metric("Confidence", result["confidence"])
    c3.metric("Grounded", str(result["grounded"]))
    c4.metric("Latency", f"{result['latency_sec']:.2f}s")
    st.subheader("Sources")
    for h in result["sources"]:
        with st.expander(f"[{h['score']:.3f}] {h['source_file']} #{h['chunk_index']}"):
            st.write(h["text"])
