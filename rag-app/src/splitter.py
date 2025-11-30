# splitter.py

import tiktoken

def split_text(text: str, chunk_size=150, overlap=30):
    """Split text into token-based chunks."""
    enc = tiktoken.get_encoding("cl100k_base")

    tokens = enc.encode(text)
    chunks = []

    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk = tokens[start:end]
        chunk_text = enc.decode(chunk)
        chunks.append(chunk_text)
        start += chunk_size - overlap

    return chunks
