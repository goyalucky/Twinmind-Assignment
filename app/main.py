from dotenv import load_dotenv
load_dotenv()

import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import SQLModel, create_engine
from pathlib import Path
import uvicorn

from .vector_store import FaissStore
from .processor import Ingestor
from .retriever import Retriever
from .llm import synthesize_answer

BASE_DIR = Path(__file__).resolve().parent
DB_FILE = BASE_DIR.parent / "data" / "db.sqlite"
DB_FILE.parent.mkdir(parents=True, exist_ok=True)

DATABASE_URL = f"sqlite:///{DB_FILE}"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SQLModel.metadata.create_all(engine)

EMBED_DIM = 1536
faiss_store = FaissStore(dim=EMBED_DIM, path=str(BASE_DIR.parent / "faiss_index"))
ingestor = Ingestor(engine, faiss_store)
retriever = Retriever(engine, faiss_store)

app = FastAPI(title="SecondBrain")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "Backend running successfully"}

@app.post("/query")
async def query(
    q: str = Form(...),
    top_k: int = Form(6)
):
    contexts = retriever.retrieve(q, top_k)

    if not contexts:
        return {
            "answer": "No relevant data found. Please ingest documents first.",
            "contexts": []
        }

    try:
        answer = synthesize_answer(q, contexts, max_tokens=400)
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

    return {
        "answer": answer,
        "contexts": contexts
    }

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
