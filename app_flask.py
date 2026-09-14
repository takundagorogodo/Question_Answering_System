"""
Stage 15 - Flask UI - NO pyarrow, NO streamlit, bypasses WDAC for UI process
UI process never imports torch or pyarrow. Child process does via cli_answer.

Run: python app_flask.py
Then open http://localhost:5000
"""
from flask import Flask, request, jsonify, render_template_string
import subprocess
import sys
import json
from pathlib import Path
import os

app = Flask(__name__)

HTML = """
<!doctype html>
<html>
<head>
<title>Generative QA - RAG</title>
<style>
body{font-family:Arial; max-width:900px; margin:20px auto; background:#f5f5f5; color:#222}
.card{background:white; padding:20px; border-radius:10px; box-shadow:0 2px 5px rgba(0,0,0,0.1); margin-bottom:15px}
.badge{display:inline-block; padding:4px 8px; border-radius:5px; font-size:12px; margin-right:5px}
.high{background:#d4edda} .medium{background:#fff3cd} .low{background:#f8d7da}
.true{background:#d4edda} .false{background:#f8d7da}
input,button{padding:10px; font-size:16px}
input{width:70%} button{width:20%; cursor:pointer}
pre{white-space:pre-wrap; background:#eee; padding:10px; border-radius:5px}
</style>
</head>
<body>
<h1>Question Answering System using NLP and Generative QA</h1>
<p>RAG + FLAN-T5-base | threshold 0.35 | CPU-only | Flask bypass (no pyarrow)</p>

<div class="card">
<form method="POST" action="/ask">
<input name="question" placeholder="Explain tokenization in simple words." value="{{q}}" required>
<button type="submit">Get Answer</button>
</form>
<p>Try: Explain tokenization | What is RAG? | What is hallucination? | What is best recipe for chocolate cake? (OOD test)</p>
</div>

{% if result %}
<div class="card">
<h3>Answer</h3>
<p><b>{{result.answer}}</b></p>
<p>
<span class="badge {{result.confidence}}">confidence: {{result.confidence}}</span>
<span class="badge {{result.grounded}}">grounded: {{result.grounded}}</span>
<span class="badge">top_score: {{result.top_score}}</span>
<span class="badge">latency: {{result.latency_sec}}s</span>
</p>
</div>

<div class="card">
<h3>Sources (auditable)</h3>
{% for s in result.sources %}
<details>
<summary>[{{"%.3f"|format(s.score)}}] {{s.source_file}} #{{s.chunk_index}}</summary>
<pre>{{s.text}}</pre>
<small>{{s.chunk_id}} | {{s.char_count}} chars</small>
</details>
{% endfor %}
</div>
{% endif %}

{% if error %}
<div class="card" style="background:#f8d7da">
<h3>Error</h3>
<pre>{{error}}</pre>
</div>
{% endif %}

</body>
</html>
"""

def get_answer_via_subprocess(question: str) -> dict:
    cmd = [sys.executable, "-m", "src.cli_answer", question]
    project_root = Path(__file__).parent
    result = subprocess.run(cmd, cwd=str(project_root), capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"Subprocess failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
    # last line is JSON
    line = result.stdout.strip().splitlines()[-1]
    return json.loads(line)

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML, result=None, q="", error=None)

@app.route("/ask", methods=["POST"])
def ask():
    q = request.form.get("question", "")
    try:
        res = get_answer_via_subprocess(q)
        return render_template_string(HTML, result=res, q=q, error=None)
    except Exception as e:
        return render_template_string(HTML, result=None, q=q, error=str(e))

if __name__ == "__main__":
    print("Starting Flask UI on http://localhost:5000 - no pyarrow, no torch in main process")
    app.run(host="0.0.0.0", port=5000, debug=False)
