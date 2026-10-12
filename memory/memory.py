import hashlib
import os
import time
import uuid
from enum import Enum

import chromadb

class MemoryType(Enum):
    EXPERIENCE = "experience"      # 经历
    IMPRESSION = "impression"      # 印象
    RELATIONSHIP = "relationship"  # 关系



class MemoryRAG:
    def __init__(self, embedding=None, db_name="memory"):
        self.embedding = embedding
        self.db_name = db_name

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.path = os.path.join(base_dir, "chromadb_saves")
        os.makedirs(self.path, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=self.path
        )

        if self.embedding is None:
            # Chroma supplies its local default embedding function.
            self.collection = self.client.get_or_create_collection(name=self.db_name)
        else:
            self.collection = self.client.get_or_create_collection(
                name=self.db_name,
                embedding_function=self.embedding,
            )

    def add_memory(
        self,
        text: str,
        memory_type: MemoryType,
        related_to: str,
        owner: str,
        updated_at: int | None = None,
        source_id: str | None = None,
    ):
        """Add an event or replace the current impression/relationship snapshot.

        Experience entries use ``source_id`` for idempotent writes. Impression and
        relationship entries have one stable ID per owner and related entity, so
        subsequent updates replace the current snapshot.
        """
        if memory_type == MemoryType.EXPERIENCE and source_id:
            key = f"{owner}\0{related_to}\0{memory_type.value}\0{source_id}"
        elif memory_type in (MemoryType.IMPRESSION, MemoryType.RELATIONSHIP):
            key = f"{owner}\0{related_to}\0{memory_type.value}"
        else:
            key = str(uuid.uuid4())
        memory_id = hashlib.sha256(key.encode("utf-8")).hexdigest()

        metadata = {
            "memory_type": memory_type.value,
            "related_to": related_to,
            "owner": owner,
            "updated_at": int(time.time()) if updated_at is None else int(updated_at),
        }
        if source_id:
            metadata["source_id"] = source_id

        self.collection.upsert(
            documents=[text],
            metadatas=[metadata],
            ids=[memory_id],
        )
        return memory_id



    def search(
        self,
        query: str,
        top_k: int = 5,
        where: dict | None = None,
    ):
        """检索相关角色记忆"""

        if top_k <= 0 or self.collection.count() == 0:
            return []

        top_k = min(top_k, self.collection.count())

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where,
        )

        return results

    def get_current_memory(
        self,
        memory_type: MemoryType,
        related_to: str,
        owner: str,
    ) -> dict | None:
        """Read the current impression or relationship snapshot, if any."""
        result = self.collection.get(
            where={"$and": [
                {"memory_type": memory_type.value},
                {"related_to": related_to},
                {"owner": owner},
            ]},
            include=["documents", "metadatas"],
        )
        documents = result.get("documents") or []
        if not documents:
            return None
        return {
            "id": result["ids"][0],
            "text": documents[0],
            "metadata": (result.get("metadatas") or [{}])[0],
        }

    
            



    def delete(self, memory_ids: list[str]):
        self.collection.delete(ids=memory_ids)
