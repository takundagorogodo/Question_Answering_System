import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from src.qa_pipeline import answer_question

st.set_page_config(page_title="Generative QA System", page_icon="🧠")

st.title("🧠 Generative Question Answering System")
st.caption("Answers are generated, not copied — grounded in retrieved sources, with visible confidence.")

question = st.text_input("Ask a question about NLP, embeddings, RAG, or transformers:")
ask_clicked = st.button("Ask")

if ask_clicked and question.strip():
    with st.spinner("Retrieving context and generating answer..."):
        result = answer_question(question)

    st.subheader("Answer")
    st.write(result["answer"])

    # Confidence / grounding indicator
    if result["grounded"]:
        st.success(f"✅ Grounded — top relevance score: {result['top_score']:.2f}")
    else:
        st.warning(f"⚠️ Not grounded — best match too weak (score: {result['top_score']:.2f})")

    # Collapsible sources
    if result["sources"]:
        with st.expander("📄 Show retrieved sources"):
            for s in result["sources"]:
                st.markdown(f"**{s['title']}** ({s['chunk_id']}) — score: {s['score']:.4f}")
                st.write(s["text"])
                st.divider()

    # Lightweight feedback
    st.write("Was this answer helpful?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍 Yes"):
            with open("evaluation/feedback_log.csv", "a", encoding="utf-8") as f:
                f.write(f'"{question}","{result["answer"]}",helpful\n')
            st.toast("Thanks for the feedback!")
    with col2:
        if st.button("👎 No"):
            with open("evaluation/feedback_log.csv", "a", encoding="utf-8") as f:
                f.write(f'"{question}","{result["answer"]}",not_helpful\n')
            st.toast("Thanks for the feedback!")

elif ask_clicked:
    st.info("Please type a question first.")