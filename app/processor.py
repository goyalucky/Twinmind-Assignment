import os
from typing import List, Optional
from datetime import datetime

import numpy as np
import aiohttp
from bs4 import BeautifulSoup
from readability import Document as ReadabilityDocument
from sqlmodel import Session

from .utils import simple_chunk_text
from .models import Document, Chunk
from .vector_store import FaissStore

# ---------------- CONFIG ---------------- #

USE_MOCK = True  # Set False in production

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_EMBED_URL = "https://api.groq.ai/v1/embeddings"
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 1536

if not USE_MOCK and not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY not found. Set it as an environment variable or in .env"
    )

# ---------------- INGESTOR ---------------- #

class Ingestor:
    def __init__(self, db_engine, faiss_store: FaissStore):
        self.engine = db_engine
        self.store = faiss_store

    # ---------- FILE INGESTION ---------- #

    async def ingest_file(
        self,
        path: str,
        source_type: str,
        title: Optional[str] = None
    ):
        if source_type in ("pdf", "md", "txt"):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            return await self._ingest_text(text, path, source_type, title)

        if source_type == "audio":
            return {
                "status": "skipped",
                "reason": "Audio ingestion not supported (ffmpeg missing)"
            }

        raise ValueError(f"Unsupported source type: {source_type}")

    # ---------- URL INGESTION ---------- #

    async def ingest_url(self, url: str, source_type: str = "web"):
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=20) as resp:
                html = await resp.text()

        doc = ReadabilityDocument(html)
        soup = BeautifulSoup(doc.summary(), "html.parser")
        text = soup.get_text(separator="\n")

        return await self._ingest_text(text, url, source_type)

    # ---------- EMBEDDINGS ---------- #

    async def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        if USE_MOCK:
            return [
                np.zeros(EMBED_DIM, dtype=np.float32).tolist()
                for _ in texts
            ]

        import requests

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": EMBED_MODEL,
            "input": texts,
        }

        resp = requests.post(
            GROQ_EMBED_URL,
            headers=headers,
            json=payload,
            timeout=30,
        )

        if resp.status_code != 200:
            raise RuntimeError(
                f"GROQ embedding API error {resp.status_code}: {resp.text}"
            )

        return [item["embedding"] for item in resp.json().get("data", [])]

    # ---------- CORE INGESTION ---------- #

    async def _ingest_text(
        self,
        text: str,
        source: str,
        source_type: str,
        title: Optional[str] = None
    ):
        text = text.strip()
        if not text:
            return {"status": "empty"}

        # Save document
        with Session(self.engine) as session:
            doc = Document(
                title=title or source,
                source=source,
                source_type=source_type,
                created_at=datetime.utcnow(),
            )
            session.add(doc)
            session.commit()
            session.refresh(doc)

        # Chunk text
        chunks = simple_chunk_text(
            text,
            max_tokens=400,
            overlap=80,
        )

        # Create embeddings
        vectors = await self._embed_texts(chunks)
        np_vectors = np.array(vectors, dtype=np.float32)

        # Metadata
        metas = [
            {
                "document_id": doc.id,
                "source": source,
                "source_type": source_type,
                "chunk_index": i,
                "created_at": datetime.utcnow().isoformat(),
            }
            for i in range(len(chunks))
        ]

        # Store in FAISS
        ids = self.store.add(np_vectors, metas)

        # Store chunks in DB
        with Session(self.engine) as session:
            for i, chunk_text in enumerate(chunks):
                session.add(
                    Chunk(
                        document_id=doc.id,
                        text=chunk_text,
                        embedding_id=ids[i],
                        meta_data=metas[i],
                        created_at=datetime.utcnow(),
                    )
                )
            session.commit()

        return {
            "status": "ok",
            "document_id": doc.id,
            "num_chunks": len(chunks),
        }
