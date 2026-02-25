QUALITY_BUCKETS = {
    "depth": ["depth"],
    "gain": ["gain"],
    "alignment": ["beam", "aligned", "perpendicular"],
    "coverage": ["required structures", "visible"],
    "artifact": ["shadow", "artifact", "noise"]
}

def summarize_guidelines(results):
    found = set()

    for r in results:
        text = r["text"].lower()

        for label, keywords in QUALITY_BUCKETS.items():
            for k in keywords:
                if k in text:
                    found.add(label)

    explanations = {
        "depth": "Image depth does not fully capture the target anatomy.",
        "gain": "Gain settings reduce contrast or obscure structures.",
        "alignment": "Probe alignment causes distortion or foreshortening.",
        "coverage": "Required structures are partially missing.",
        "artifact": "Artifacts or shadowing reduce interpretability."
    }

    return [explanations[f] for f in found]

import json
import numpy as np
import os
from numpy.linalg import norm

ROOT = os.path.dirname(os.path.abspath(__file__))
chunks_path = os.path.join(ROOT, "chunks.json")
emb_path = os.path.join(ROOT, "embeddings_local.json")

if not os.path.exists(chunks_path):
    raise FileNotFoundError(f"{chunks_path} not found")
if not os.path.exists(emb_path):
    raise FileNotFoundError(f"{emb_path} not found — run embed_chunks.py first")

# Load chunks
with open(chunks_path, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Load embeddings
with open(emb_path, "r", encoding="utf-8") as f:
    data = json.load(f)

embeddings = np.array([np.array(item["embedding"], dtype=float) for item in data])

def _build_embedder():
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Using sentence-transformers for query encoding")
        return lambda text: model.encode(text)
    except Exception as e:
        print("sentence-transformers not available, falling back to transformers+torch:", e)
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            hf_model_name = "sentence-transformers/all-MiniLM-L6-v2"
            tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
            hf_model = AutoModel.from_pretrained(hf_model_name)
            print("Using transformers+torch for query encoding")

            def embed_text(text):
                inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                with torch.no_grad():
                    out = hf_model(**inputs, return_dict=True)
                    last = out.last_hidden_state
                    mask = inputs["attention_mask"].unsqueeze(-1).expand(last.size()).float()
                    summed = torch.sum(last * mask, 1)
                    counts = torch.clamp(mask.sum(1), min=1e-9)
                    mean_pooled = (summed / counts)[0]
                    return mean_pooled.cpu().numpy()

            return embed_text
        except Exception as e2:
            raise RuntimeError("No embedding backend available. Install `sentence-transformers` or `transformers[torch]`.") from e2

embed_text = None

def cosine_similarity(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    denom = (norm(a) * norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def query_rag(question, top_k=3):
    global embed_text
    if embed_text is None:
        embed_text = _build_embedder()
    query_embedding = embed_text(question)

    scores = []
    for i, emb in enumerate(embeddings):
        score = cosine_similarity(query_embedding, emb)
        scores.append((score, i))

    scores.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, idx in scores[:top_k]:
        results.append({
            "score": float(score),
            "text": chunks[idx]["text"]
        })

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RAG query over local chunks")
    parser.add_argument("--question", "-q", help="Question text to query (if omitted, prompt interactively)")
    parser.add_argument("--top_k", "-k", type=int, default=3, help="Number of top matches to return")
    parser.add_argument("--quiet", action="store_true", help="Reduce TF/transformers logging (set before loading backend)")
    args = parser.parse_args()

    if args.quiet:
        # minimize TF logs before building embedder
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    if args.question:
        question = args.question
    else:
        try:
            # try reading from stdin (pipe) first
            import sys
            if not sys.stdin.isatty():
                question = sys.stdin.read().strip()
            else:
                question = input("Ask a question about ultrasound quality: ")
        except Exception:
            question = input("Ask a question about ultrasound quality: ")

    results = query_rag(question, top_k=args.top_k)

    print("\nTop matches:\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. Score: {r['score']:.3f}")
        print(r["text"])
        print()

    # Provide guideline summary if available
    try:
        summary = summarize_guidelines(results)
        if summary:
            print("Guideline suggestions:")
            for s in summary:
                print("-", s)
    except Exception:
        pass
