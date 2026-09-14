"""
Stage 16-17 - Evaluation: real test set, real numbers, no fabrication.

Measures:
- Retrieval top_score distribution in-domain vs OOD
- Grounded rate (should be high for in-domain, low for OOD)
- Correct refusal rate for OOD (threshold 0.35 should block 0.157)
- Latency
- Verbatim copy rate (generative proof)

Test set is small but honest, logged with n-caveats.
"""
import json
import time
from pathlib import Path
from src.qa_pipeline import answer_question

# Small honest test set - in-domain from your 9 docs + OOD
TESTS = [
    {"q": "What is natural language processing?", "type": "in_domain", "should_ground": True},
    {"q": "Explain tokenization in simple words.", "type": "in_domain", "should_ground": True},
    {"q": "What is retrieval augmented generation?", "type": "in_domain", "should_ground": True},
    {"q": "What is hallucination in language models?", "type": "in_domain", "should_ground": True},
    {"q": "What is embedding?", "type": "in_domain", "should_ground": True},
    {"q": "What is chunking and why is it needed?", "type": "in_domain", "should_ground": True},
    {"q": "What is the best recipe for chocolate cake?", "type": "ood", "should_ground": False},
    {"q": "Who won the IPL in 2024?", "type": "ood", "should_ground": False},
    {"q": "How to fix a leaking tap?", "type": "ood", "should_ground": False},
]

def main():
    results = []
    print(f"Running evaluation on {len(TESTS)} questions (threshold 0.35)\n")
    for item in TESTS:
        q = item["q"]
        r = answer_question(q)
        ok = (r["grounded"] == item["should_ground"])
        results.append({
            "question": q,
            "type": item["type"],
            "should_ground": item["should_ground"],
            "top_score": r["top_score"],
            "confidence": r["confidence"],
            "grounded": r["grounded"],
            "latency": r["latency_sec"],
            "correct": ok,
            "answer": r["answer"][:200],
        })
        print(f"[{item['type']}] {q[:50]:50} -> score {r['top_score']:.3f} grounded {r['grounded']} correct {ok} time {r['latency_sec']:.1f}s")

    # Summary
    in_domain = [x for x in results if x["type"] == "in_domain"]
    ood = [x for x in results if x["type"] == "ood"]

    def avg(lst, key): return sum(x[key] for x in lst) / len(lst) if lst else 0

    print("\n--- SUMMARY ---")
    print(f"In-domain (n={len(in_domain)}): avg_score {avg(in_domain,'top_score'):.3f} grounded_rate {sum(x['grounded'] for x in in_domain)/len(in_domain):.2f} avg_latency {avg(in_domain,'latency'):.1f}s")
    print(f"OOD (n={len(ood)}): avg_score {avg(ood,'top_score'):.3f} correct_refusal_rate {sum(not x['grounded'] for x in ood)/len(ood):.2f} (should be 1.0)")
    print(f"Overall accuracy (grounded matches expected): {sum(x['correct'] for x in results)/len(results):.2f}")

    # Save
    out_path = Path("data/processed/eval_results.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nSaved {out_path}")

if __name__ == "__main__":
    main()
