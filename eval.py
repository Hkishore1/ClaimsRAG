import json
import requests
import os
from dotenv import load_dotenv

load_dotenv()

EVAL_FILE = os.getenv("EVAL_FILE", "eval.jsonl")
ASK_API_URL = os.getenv("ASK_API_URL", "http://localhost:8000/ask")

def evaluate():
    queries = []
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line))
    
    total = len(queries)
    hits = 0
    precision_sum = 0

    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)

    for i, q in enumerate(queries, 1):
        payload = {"query": q["q"], "k": 3}
        resp = requests.post(ASK_API_URL, json=payload).json()
        citations = resp.get("citations", [])
        
        # Case-insensitive, whitespace-normalized matching
        expected = q["ans_contains"].lower().strip()
        correct = sum(1 for c in citations 
                     if expected in c.get("full_snippet", "").lower())
        
        if correct > 0:
            hits += 1
            precision_sum += correct / 3

        # Detailed output
        print(f"\n[{i}] Query: {q['q']}")
        print(f"    Expected: '{q['ans_contains']}'")
        print(f"    Matches: {correct}/3")
        print(f"    Status: {'✓ HIT' if correct > 0 else '✗ MISS'}")
        
        # Debug misses
        if correct == 0:
            print("    Retrieved snippets:")
            for j, c in enumerate(citations, 1):
                print(f"      [{j}] {c['doc']}: {c.get('snippet', '')[:60]}...")

    hit_rate = hits / total
    precision_at_k = precision_sum / total
    
    print("\n" + "="*80)
    print(f"SUMMARY: n={total} | hit_rate={hit_rate:.2f} | precision@3={precision_at_k:.2f}")
    print("="*80 + "\n")

if __name__ == "__main__":
    evaluate()