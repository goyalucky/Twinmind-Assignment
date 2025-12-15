import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Read API key securely from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY not found. Please set it in the .env file or environment variables."
    )

# Groq API endpoint
GROQ_API_URL = "https://api.groq.ai/v1/completions"

# System prompt
SYSTEM_PROMPT = (
    'You are a helpful AI "Second Brain" assistant. '
    "Use the provided context snippets to answer concisely and helpfully. "
    "If the answer is not present in the context, say you don't know."
)


def synthesize_answer(query: str, contexts: list, max_tokens: int = 256) -> str:
    """
    Sends a request to the Groq API and returns the generated answer.
    """

    context_text = "\n\n---\n\n".join(
        f"Source: {c.get('source', 'Unknown')}\nText: {c.get('text', '')}"
        for c in contexts
    )

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context:\n{context_text}\n\n"
        f"User query: {query}\n\n"
        f"Answer:"
    )

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "gpt-4o-mini",
        "input": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }

    response = requests.post(
        GROQ_API_URL,
        headers=headers,
        json=payload,
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"GROQ API Error {response.status_code}: {response.text}"
        )

    data = response.json()
    return data.get("output", "").strip()
