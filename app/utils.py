# backend/app/utils.py
import re
from typing import List

def simple_chunk_text(text: str, max_tokens: int = 500, overlap: int = 50) -> List[str]:
    """
    Very simple chunker based on words, not tokens.
    max_tokens ~ words for simplicity.
    """
    words = re.split(r"\s+", text.strip())
    chunks = []
    i = 0
    n = len(words)
    while i < n:
        j = min(i + max_tokens, n)
        chunk = " ".join(words[i:j])
        chunks.append(chunk)
        i = j - overlap
        if i < 0:
            i = 0
    return chunks
