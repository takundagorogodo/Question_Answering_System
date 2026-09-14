"""
Stage 15 - Streamlit UI - subprocess bypass for WDAC torch block
Streamlit process NEVER imports torch. Child python.exe does.

Run: streamlit run app.py --server.fileWatcherType none
"""
import streamlit as st
import subprocess
import sys
import json
from pathlib import Path

from src.config import RETRIEVAL_TOP_K, SIMILARITY_THRESHOLD

st.set_page_config(page_title="Generative QA - RAG", layout="wide")
st.title("Question Answering System using NLP and Generative QA")
st.caption(f"RAG + FLAN-T5-base | top_k={RETRIEVAL_TOP_K} | threshold={SIMILARITY_THRESHOLD} | CPU-only | WDAC bypass via subprocess")

with st.sidebar:
    st.header("How it works")
    st.markdown("""
    1. Question -> embedding (MiniLM-L6-v2, 384-dim)
    2. FAISS cosine -> top-K + scores
    3. If top_score < threshold -> refuse
    4. Else FLAN-T5-base generates (in child process)
    
    **WDAC fix:** Streamlit never imports torch. Child `python.exe` does (same as `python -m src.qa_pipeline` which worked).
    """)
    st.metric("Threshold", SIMILARITY_THRESHOLD)
    st.metric("Top-K", RETRIEVAL_TOP_K)

examples = [
    "Explain tokenization in simple words.",
    "What is retrieval augmented generation?",
    "What is hallucination in language models?",
    "What is the best recipe for chocolate cake?",
    "explain lemmatization",
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

def get_answer_via_subprocess(question: str) -> dict:
    """Call cli_answer.py in a separate python.exe process"""
    # Use same python executable as this venv
    cmd = [sys.executable, "-m", "src.cli_answer", question]
    # cwd = project root
    project_root = Path(__file__).parent
    result = subprocess.run(cmd, cwd=str(project_root), capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        # Include stderr for debugging
        raise RuntimeError(f"Subprocess failed (code {result.returncode}):\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
    # stdout should be JSON
    try:
        data = json.loads(result.stdout.strip().splitlines()[-1])  # last line is JSON
        return data
    except Exception as e:
        raise RuntimeError(f"Failed to parse JSON: {e}\nRAW: {result.stdout}\nERR: {result.stderr}")

if st.button("Get Answer") and q:
    with st.spinner("Child process: retrieving + generating... first ~15-20s, next ~3-6s"):
        try:
            res = get_answer_via_subprocess(q)
        except Exception as e:
            st.error("Model process failed")
            st.code(str(e))
            st.stop()

    st.subheader("Answer")
    if res.get("grounded"):
        st.success(res["answer"])
    else:
        st.warning(res["answer"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Top Score", f"{res['top_score']:.3f}")
    c2.metric("Confidence", res["confidence"])
    c3.metric("Grounded", str(res["grounded"]))
    c4.metric("Latency", f"{res['latency_sec']:.2f}s")

    st.subheader("Sources (auditable)")
    for h in res["sources"]:
        with st.expander(f"[{h['score']:.3f}] {h['source_file']} #{h['chunk_index']}"):
            st.write(h["text"])
            st.caption(f"{h['chunk_id']} | {h['char_count']} chars")
else:
    st.info("Enter question and click Get Answer. OOD example should refuse.")
