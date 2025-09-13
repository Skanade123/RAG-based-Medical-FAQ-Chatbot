# src/build_index.py

import os
import json
import numpy as np
from tqdm import tqdm
import faiss
import google.generativeai as genai
from utils import load_dataset, chunk_text


# Ask user for API key if not found in environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    GEMINI_API_KEY = input("🔑 Enter your GEMINI API key: ").strip()

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Embedding model
EMB_MODEL = "models/text-embedding-004"


def build_index(
    dataset_path: str,
    index_path: str = "faiss_index.bin",
    meta_path: str = "index_meta.json",
    limit: int = 100  # ✅ new param to restrict entries
):
    # Load records from dataset
    records = load_dataset(dataset_path)

    # Take only first 100 entries (or less if dataset smaller)
    records = records[:limit]
    print(f"⚡ Using only first {len(records)} records from dataset")

    # Prepare chunks
    docs = []
    for rec in records:
        combined_text = f"Q: {rec['question']}\nA: {rec['answer']}"
        chunks = chunk_text(combined_text, max_chars=800, overlap=150)
        for i, chunk in enumerate(chunks):
            docs.append({
                "doc_id": f"{rec['id']}_chunk{i}",
                "text": chunk,
                "origin_id": rec['id'],
                "title": rec['title'],
                "source": rec.get('source', '')
            })

    print(f"Total chunks to embed: {len(docs)}")

    # Embed texts (Gemini free tier: one by one)
    texts = [d["text"] for d in docs]
    vectors = []

    print("Embedding texts with Gemini...")
    for text in tqdm(texts, desc="Embedding"):
        try:
            result = genai.embed_content(
                model=EMB_MODEL,
                content=text,
                task_type="retrieval_document"
            )
            vectors.append(result['embedding'])
        except Exception as e:
            print(f"❌ Error embedding text: {e}")
            # Use a zero vector as fallback
            vectors.append([0.0] * 768)

    # Convert to numpy
    vectors = np.array(vectors).astype("float32")

    # Normalize for cosine similarity
    faiss.normalize_L2(vectors)

    # Create FAISS index (cosine similarity)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    # Save index
    faiss.write_index(index, index_path)

    # Save metadata
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    print(f"✅ Index saved to {index_path}, metadata saved to {meta_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=r"D:\AI Projects\RAG-based Medical FAQ Chatbot\data\train.csv")
    p.add_argument("--index", default="faiss_index.bin")
    p.add_argument("--meta", default="index_meta.json")
    p.add_argument("--limit", type=int, default=100, help="Number of records to use (default=100)")
    args = p.parse_args()

    build_index(args.data, args.index, args.meta, args.limit)
