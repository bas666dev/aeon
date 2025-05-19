# aeon_whisper_data.py

import os
from typing import Optional, List
from datetime import datetime
from bson import ObjectId
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorClient
from neo4j import AsyncGraphDatabase

# ---- Whisper Model ----

class WhisperModel(BaseModel):
    id: Optional[str] = Field(alias="_id")
    raw_text: str
    source_url: Optional[str]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    embeddings: List[float]
    tags: List[str] = []
    graph_id: Optional[str]  # Neo4j node ID

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {datetime: lambda v: v.isoformat()}

# ---- Repository ----

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "password")
DB_NAME = os.getenv("MONGO_DBNAME", "aeon")

class WhisperRepository:
    def __init__(self):
        self.mongo_client = AsyncIOMotorClient(MONGO_URI)
        self.mongo = self.mongo_client[DB_NAME]
        self.neo4j = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
        self._indexes_created = False

    async def ensure_indexes(self):
        if not self._indexes_created:
            await self.mongo.whispers.create_index("created_at")
            await self.mongo.whispers.create_index("tags")
            self._indexes_created = True

    async def create(self, whisper: WhisperModel) -> WhisperModel:
        await self.ensure_indexes()
        doc = whisper.dict(by_alias=True, exclude={"id"})
        result = await self.mongo.whispers.insert_one(doc)
        whisper.id = str(result.inserted_id)
        # Upsert into Neo4j
        async with self.neo4j.session() as session:
            await session.run(
                """
                MERGE (w:Whisper {mongo_id: $id})
                SET w.raw_text = $raw_text, w.source_url = $source_url, w.created_at = $created_at
                WITH w
                UNWIND $tags AS tag
                MERGE (t:Tag {value: tag})
                MERGE (w)-[:TAGGED]->(t)
                """,
                id=whisper.id,
                raw_text=whisper.raw_text,
                source_url=whisper.source_url,
                created_at=whisper.created_at.isoformat(),
                tags=whisper.tags or [],
            )
        return whisper

    async def get(self, id: str) -> Optional[WhisperModel]:
        await self.ensure_indexes()
        try:
            doc = await self.mongo.whispers.find_one({"_id": ObjectId(id)})
        except Exception:
            return None
        return WhisperModel(**doc) if doc else None

    async def update(self, id: str, **fields) -> Optional[WhisperModel]:
        await self.ensure_indexes()
        if not fields:
            return await self.get(id)
        await self.mongo.whispers.update_one({"_id": ObjectId(id)}, {"$set": fields})
        updated = await self.get(id)
        # Only update tags if tags field is present and not None
        if "tags" in fields and fields["tags"] is not None:
            async with self.neo4j.session() as session:
                await session.run(
                    """
                    MATCH (w:Whisper {mongo_id: $id})-[r:TAGGED]->(t:Tag)
                    DELETE r
                    WITH w
                    UNWIND $tags AS tag
                    MERGE (t:Tag {value: tag})
                    MERGE (w)-[:TAGGED]->(t)
                    """,
                    id=id,
                    tags=fields["tags"],
                )
        return updated

    async def delete(self, id: str) -> bool:
        await self.ensure_indexes()
        result = await self.mongo.whispers.delete_one({"_id": ObjectId(id)})
        async with self.neo4j.session() as session:
            await session.run(
                "MATCH (w:Whisper {mongo_id: $id}) DETACH DELETE w",
                id=id,
            )
        return result.deleted_count == 1

    async def aclose(self):
        self.mongo_client.close()
        await self.neo4j.close()

# ---- Ingest Script Example ----

import sys
import asyncio
from pathlib import Path

def get_embeddings(text: str):
    # TODO: Replace with real embedding model, e.g., OpenAI 'text-embedding-3-small'
    return [0.0] * 384

async def ingest_file(repo, path: Path):
    try:
        text = path.read_text(encoding="utf-8")
        whisper = WhisperModel(
            raw_text=text,
            source_url=str(path),
            embeddings=get_embeddings(text),
            tags=[],
        )
        await repo.create(whisper)
        print(f"Ingested {path.name}")
    except Exception as e:
        print(f"⚠️  {path.name} failed: {e}")

async def ingest_folder(folder):
    repo = WhisperRepository()
    files = list(Path(folder).glob("*.md")) + list(Path(folder).glob("*.txt"))
    tasks = [ingest_file(repo, f) for f in files]
    await asyncio.gather(*tasks)
    await repo.aclose()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python aeon_whisper_data.py path/to/files/")
        sys.exit(1)
    asyncio.run(ingest_folder(sys.argv[1]))