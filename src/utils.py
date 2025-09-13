# src/utils.py

import re
import pandas as pd
from typing import List, Dict

def validate_record(record: Dict) -> bool:
    """
    Validate a single dataset record.
    Ensures fields are not empty or suspicious.
    """
    if not record.get("question") or not record.get("answer"):
        return False

    # Simple sanitization: block malicious instructions inside dataset
    forbidden_patterns = [
        r"ignore\s+previous", r"system\s*prompt", r"reveal",
        r"jailbreak", r"override", r"api\s*key", r"password"
    ]
    text = (record.get("question", "") + " " + record.get("answer", "")).lower()
    for pattern in forbidden_patterns:
        if re.search(pattern, text):
            return False
    return True


def load_dataset(path: str, limit: int = 100) -> List[Dict]:
    """
    Load and validate the medical FAQ dataset from CSV (qtype, Question, Answer).
    Returns at most `limit` validated entries.
    """
    df = pd.read_csv(path)

    # Standardize column names
    df.columns = [c.strip().lower() for c in df.columns]

    required_cols = ["qtype", "question", "answer"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Dataset missing column '{col}'. Found: {df.columns.tolist()}")

    records = []
    for idx, row in df.iterrows():
        record = {
            "id": str(idx),                # unique ID
            "title": str(row["qtype"]),    # category/type
            "question": str(row["question"]).strip(),
            "answer": str(row["answer"]).strip(),
            "source": ""
        }

        # Only keep if valid
        if validate_record(record):
            records.append(record)

        # Stop if we reach the limit
        if len(records) >= limit:
            break

    return records


def chunk_text(text: str, max_chars: int = 800, overlap: int = 100) -> List[str]:
    """
    Split long text into overlapping chunks.
    Attempts to break at sentence boundaries for readability.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        if end >= len(text):
            chunks.append(text[start:].strip())
            break

        segment = text[start:end]

        # Cut at last sentence-ending punctuation if possible
        match = re.search(r'[.!?](?!.*[.!?])', segment)
        if match:
            cutoff = start + match.end()
            if cutoff - start < max_chars * 0.3:  # avoid tiny chunks
                cutoff = end
        else:
            cutoff = end

        chunk = text[start:cutoff].strip()
        if chunk:
            chunks.append(chunk)

        # Move start forward with overlap
        start = max(cutoff - overlap, cutoff)

    return chunks


# Quick test when running directly
if __name__ == "__main__":
    dataset_path = r"D:\AI Projects\RAG-based Medical FAQ Chatbot\data\train.csv"
    records = load_dataset(dataset_path, limit=100)
    print(f"✅ Loaded {len(records)} validated records. Example:")
    print(records[0])

    # Test chunking
    example_text = records[0]["answer"]
    chunks = chunk_text(example_text, max_chars=200, overlap=50)
    print(f"\nAnswer split into {len(chunks)} chunks:")
    for i, c in enumerate(chunks):
        print(f"[Chunk {i}] {c}\n")
