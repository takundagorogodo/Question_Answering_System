"""
CLI helper for Streamlit bypass: called via subprocess so torch loads in
a plain python.exe process (which worked for qa_pipeline), not inside
Streamlit's process which WDAC blocks.

Usage: python -m src.cli_answer "your question"
Outputs JSON to stdout.
"""
import sys
import json

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "no question provided"}))
        return
    question = " ".join(sys.argv[1:])

    # Import here - this is the process that needs torch
    from src.qa_pipeline import answer_question

    result = answer_question(question)
    # Make sources JSON serializable (they already are)
    # Convert any non-serializable
    out = {
        "question": result["question"],
        "answer": result["answer"],
        "top_score": result["top_score"],
        "confidence": result["confidence"],
        "grounded": result["grounded"],
        "latency_sec": result["latency_sec"],
        "sources": result["sources"],
    }
    print(json.dumps(out, ensure_ascii=False))

if __name__ == "__main__":
    main()
