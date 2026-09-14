"""
Stage 15 - Streamlit UI - lazy torch load to bypass WDAC DLL block
Run from root: streamlit run app.py --server.fileWatcherType none
"""
import streamlit as st

# Config does NOT import torch - safe to import at top
from src.config import RETRIEVAL_TOP_K, SIMILARITY_THRESHOLD

st.set_page_config(page_title="Generative QA - RAG", layout="wide")
st.title("Question Answering System using NLP and Generative QA")
st.caption(f"RAG + FLAN-T5-base | top_k={RETRIEVAL_TOP_K} | threshold={SIMILARITY_THRESHOLD} | CPU-only offline")

with st.sidebar:
    st.header("How it works")
    st.markdown("""
    1. Question -> embedding (all-MiniLM-L6-v2)
    2. FAISS cosine search -> top-K + scores
    3. If top_score < threshold -> refuse (no hallucination)
    4. Else context + question -> FLAN-T5-base generates
    """)
    st.metric("Threshold", SIMILARITY_THRESHOLD)
    st.metric("Top-K", RETRIEVAL_TOP_K)
    st.info("First answer ~15s (model load), next ~3-6s on i7-1355U")

examples = [
    "Explain tokenization in simple words.",
    "What is retrieval augmented generation?",
    "What is hallucination in language models?",
    "What is the best recipe for chocolate cake?",
]

col1, col2 = st.columns([3,1])
with col1:
    q = st.text_input("Ask a question about NLP:", placeholder="e.g. Explain tokenization in simple words.")
with col2:
    st.write("")
    st.write("")
    use_example = st.selectbox("Try example:", [""] + examples)

if use_example:
    q = use_example

def load_pipeline():
    # Import HERE - after Streamlit server started, avoids WDAC block at startup
    from src.qa_pipeline import answer_question
    return answer_question

if st.button("Get Answer") and q:
    with st.spinner("Loading models + retrieving + generating... first time ~15s"):
        try:
            answer_fn = load_pipeline()
            result = answer_fn(q)
        except Exception as e:
            st.error(f"Model load failed: {e}")
            st.code(str(e))
            st.stop()

    st.subheader("Answer")
    if result["grounded"]:
        st.success(result["answer"])
    else:
        st.warning(result["answer"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Top Score", f"{result['top_score']:.3f}")
    c2.metric("Confidence", result["confidence"])
    c3.metric("Grounded", str(result["grounded"]))
    c4.metric("Latency", f"{result['latency_sec']:.2f}s")

    st.subheader("Sources (auditable)")
    for h in result["sources"]:
        with st.expander(f"[{h['score']:.3f}] {h['source_file']} #{h['chunk_index']}"):
            st.write(h["text"])
            st.caption(f"chunk_id: {h['chunk_id']} | char_count: {h['char_count']}")
else:
    st.info("Enter a question and click Get Answer. Try OOD example to see hallucination control.")
