# backend/app/models.py
from typing import Optional
from sqlmodel import SQLModel, Field, Column, JSON
from datetime import datetime
import uuid

def gen_uuid():
    return str(uuid.uuid4())

class Document(SQLModel, table=True):
    id: str = Field(default_factory=gen_uuid, primary_key=True)
    title: Optional[str]
    source: Optional[str]  
    source_type: str  
    created_at: datetime = Field(default_factory=datetime.utcnow)
    meta_data: Optional[dict] = Field(default=None, sa_column=Column(JSON))

class Chunk(SQLModel, table=True):
    id: str = Field(default_factory=gen_uuid, primary_key=True)
    document_id: str = Field(index=True)
    text: str
    start_time: Optional[float] = None  
    end_time: Optional[float] = None
    embedding_id: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    meta_data: Optional[dict] = Field(default=None, sa_column=Column(JSON))
