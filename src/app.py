"""
Stage 15 - Streamlit UI: calls qa_pipeline.answer_question()
Run: streamlit run app.py

What judges see:
- Question box
- Answer (generative)
- Confidence: high/medium/low from retrieval score
- Grounded: True/False (threshold gate)
- Sources with scores (auditable)
- Latency

Offline, CPU-only, no API key.
"""
import streamlit as st
from src.qa_pipeline import answer_question
from src.config import RETRIEVAL_TOP_K, SIMILARITY_THRESHOLD

st.set_page_config(page_title="Generative QA - RAG", layout="wide")

st.title("Question Answering System using NLP and Generative QA")
st.caption(f"RAG + FLAN-T5-base | top_k={RETRIEVAL_TOP_K} | threshold={SIMILARITY_THRESHOLD} | CPU-only offline")

# Sidebar - transparency
with st.sidebar:
    st.header("How it works")
    st.markdown("""
    **Pipeline:**
    1. Question -> embedding (all-MiniLM-L6-v2, 384-dim)
    2. FAISS cosine search -> top-K chunks + scores
    3. If top_score < threshold -> refuse (no hallucination)
    4. Else context + question -> FLAN-T5-base generates answer
    
    **Differentiator:**
    - Shows sources + scores
    - Shows confidence + grounded flag
    - Measured threshold from real scores
    """)
    st.divider()
    st.metric("Threshold", SIMILARITY_THRESHOLD)
    st.metric("Top-K", RETRIEVAL_TOP_K)

# Demo questions from your probe
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

if st.button("Get Answer") and q:
    with st.spinner("Retrieving + generating (CPU)... first answer ~15s for model load, next ~3-6s"):
        result = answer_question(q)

    # Answer block
    st.subheader("Answer")
    if result["grounded"]:
        st.success(result["answer"])
    else:
        st.warning(result["answer"])

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Top Score", f"{result['top_score']:.3f}")
    c2.metric("Confidence", result["confidence"])
    c3.metric("Grounded", str(result["grounded"]))
    c4.metric("Latency", f"{result['latency_sec']:.2f}s")

    # Sources - auditable
    st.subheader("Sources (auditable)")
    for i, h in enumerate(result["sources"]):
        with st.expander(f"[{h['score']:.3f}] {h['source_file']} #{h['chunk_index']}"):
            st.write(h["text"])
            st.caption(f"chunk_id: {h['chunk_id']} | char_count: {h['char_count']}")

    st.divider()
    st.caption("Generative check: answer is written token-by-token by FLAN-T5-base, not copied as a span. Sources shown for verification.")

else:
    st.info("Enter a question and click Get Answer. Try the OOD example to see hallucination control.")
