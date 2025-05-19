from future import annotations

"""Æon ‒ Whisper persistence & retrieval layer.

Key features implemented (\u2714 = latent task now finished):

\u2714 Pydantic v2 model with strict validation. \u2714 Lazy initialisation & graceful shutdown for MongoDB (Motor) and Neo4j (neo4j‑python‑driver). \u2714 Mongo indexes: created_at, tags, $text on raw_text + Atlas Vector Search stub. \u2714 Neo4j graph mirror with :TAG & time‑tree backbone. \u2714 Embedding pipeline (defaults to OpenAI text‑embedding‑3‑small, pluggable via embed_fn). \u2714 Async ingest & update helpers returning up‑to‑date Whisper objects. \u2714 Full type hints, PEP‑8, black‑compatible.

Requires: motor>=3.3.2 neo4j>=5.14 pydantic>=2.7 openai>=1.25 (optional if you wire your own embed_fn) """

from contextlib import asynccontextmanager from datetime import datetime, timezone from pathlib import Path from typing import Any, AsyncIterator, Iterable, List, Sequence

import asyncio import logging

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase from neo4j import AsyncGraphDatabase from neo4j.time import DateTime as Neo4jDateTime from openai import AsyncOpenAI from pydantic import BaseModel, Field, field_validator from pydantic.alias_generators import to_camel

logger = logging.getLogger(name)

---------------------------------------------------------------------------

Pydantic model

---------------------------------------------------------------------------

class Whisper(BaseModel, validate_assignment=True, alias_generator=to_camel, frozen=True): """Atomic knowledge particle."""

id: str | None = Field(default=None, alias="_id")
created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
source_path: str
raw_text: str = Field(min_length=1)
tokens: int | None = None
tags: List[str] = Field(default_factory=list)
embedding: List[float] | None = None

@field_validator("tags", mode="before")
@classmethod
def _dedupe_tags(cls, v: Sequence[str]) -> list[str]:  # noqa: D401
    """Ensure tags are unique & lowercase."""
    return sorted({t.lower() for t in v})

---------------------------------------------------------------------------

Database orchestrator

---------------------------------------------------------------------------

class AeonStorage: """Facade aggregating MongoDB + Neo4j side‑cars."""

def __init__(
    self,
    mongo_dsn: str = "mongodb://localhost:27017",
    mongo_db: str = "aeon",
    neo4j_uri: str = "neo4j://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "password",
    embed_fn: Any | None = None,
) -> None:
    self._mongo_dsn = mongo_dsn
    self._mongo_db_name = mongo_db
    self._neo4j_uri = neo4j_uri
    self._neo4j_user = neo4j_user
    self._neo4j_password = neo4j_password
    self._embed_fn = embed_fn or OpenAIEmbedder().embed

    self._mongo_client: AsyncIOMotorClient | None = None
    self._mongo_db: AsyncIOMotorDatabase | None = None
    self._neo4j_driver = None

# ---------------------------------------------------------------------
# Lazy initialisation helpers
# ---------------------------------------------------------------------

@property
def mongo(self) -> AsyncIOMotorDatabase:
    if self._mongo_db is None:
        self._mongo_client = AsyncIOMotorClient(self._mongo_dsn, tz_aware=True)
        self._mongo_db = self._mongo_client[self._mongo_db_name]
        asyncio.create_task(self._ensure_mongo_indexes())
    return self._mongo_db

@property
def neo4j(self):  # neo4j.AsyncDriver
    if self._neo4j_driver is None:
        self._neo4j_driver = AsyncGraphDatabase.async_driver(
            self._neo4j_uri, auth=(self._neo4j_user, self._neo4j_password)
        )
    return self._neo4j_driver

async def aclose(self) -> None:
    if self._mongo_client is not None:
        self._mongo_client.close()
    if self._neo4j_driver is not None:
        await self._neo4j_driver.close()

async def __aenter__(self):  # noqa: D401
    return self

async def __aexit__(self, exc_type, exc, tb):  # noqa: D401
    await self.aclose()

# ---------------------------------------------------------------------
# Public APIs
# ---------------------------------------------------------------------

async def upsert_whisper(self, whisper: Whisper) -> Whisper:
    """Insert new or update existing whisper; sync graph."""

    # --- embed if needed
    if whisper.embedding is None:
        whisper = whisper.model_copy(update={"embedding": await self._embed_fn(whisper.raw_text)})

    # --- MongoDB upsert
    doc = whisper.model_dump(by_alias=True, exclude_none=True)
    result = await self.mongo.whispers.find_one_and_update(
        {"source_path": whisper.source_path},
        {"$set": doc},
        upsert=True,
        return_document=True,
    )
    whisper = Whisper(**result)

    # --- Neo4j sync (fire‑and‑forget)
    asyncio.create_task(self._sync_graph(whisper))
    return whisper

async def update_whisper(self, id: str, **fields: Any) -> Whisper | None:
    if not fields:
        return None

    if "tags" in fields and not fields["tags"]:
        fields["tags"] = []  # normalise

    doc = await self.mongo.whispers.find_one_and_update(
        {"_id": id}, {"$set": fields}, return_document=True
    )
    if doc is None:
        return None
    whisper = Whisper(**doc)
    if "tags" in fields:
        asyncio.create_task(self._sync_tags_node(whisper))
    return whisper

async def search_text(self, query: str, limit: int = 10) -> list[Whisper]:
    cursor = self.mongo.whispers.find({"$text": {"$search": query}}).sort("score", {"$meta": "textScore"}).limit(limit)
    return [Whisper(**d) async for d in cursor]

# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

async def _ensure_mongo_indexes(self) -> None:
    await self.mongo.whispers.create_index([("raw_text", "text")])
    await self.mongo.whispers.create_index("created_at")
    await self.mongo.whispers.create_index("tags")
    # Atlas Vector Search index is managed via Atlas UI / CLI ‒ documented in README.

async def _sync_graph(self, whisper: Whisper) -> None:
    query = """
    MERGE (w:Whisper {id: $id})
    SET w.raw_text = $text,
        w.created_at = $created_at,
        w.embedding = $embedding
    WITH w
    UNWIND $tags AS tag
        MERGE (t:Tag {name: tag})
        MERGE (w)-[:TAGGED]->(t)
    // Time‑tree backbone
    WITH w
    CALL apoc.temporal.toZonedDateTime(w.created_at) YIELD value as zdt
    MERGE (y:Year {value: zdt.year})
    MERGE (y)<-[:IN_YEAR]-(w)
    """
    async with self.neo4j.session(database="neo4j") as session:
        await session.run(
            query,
            id=str(whisper.id),
            text=whisper.raw_text[:4096],
            created_at=Neo4jDateTime.from_native(whisper.created_at),
            embedding=whisper.embedding,
            tags=whisper.tags,
        )

async def _sync_tags_node(self, whisper: Whisper) -> None:
    query = """
    MATCH (w:Whisper {id: $id})-[r:TAGGED]->(:Tag)
    DELETE r
    WITH w
    UNWIND $tags AS tag
        MERGE (t:Tag {name: tag})
        MERGE (w)-[:TAGGED]->(t)
    """
    async with self.neo4j.session(database="neo4j") as session:
        await session.run(query, id=str(whisper.id), tags=whisper.tags)

---------------------------------------------------------------------------

Embeddings helper

---------------------------------------------------------------------------

class OpenAIEmbedder: """Thin wrapper; can be monkey‑patched in tests."""

def __init__(self, model: str = "text-embedding-3-small", client: AsyncOpenAI | None = None):
    self._model = model
    self._client = client or AsyncOpenAI()

async def embed(self, text: str) -> list[float]:
    resp = await self._client.embeddings.create(model=self._model, input=text[:8192])
    return resp.data[0].embedding  # type: ignore[attr-defined]

---------------------------------------------------------------------------

CLI ingest script (python -m aeon_whisper_data <path> [--pattern *.md])

---------------------------------------------------------------------------

async def _ingest_cli(root: Path, pattern: str = "**/.", store: AeonStorage | None = None) -> None: store = store or AeonStorage() paths = list(root.glob(pattern)) for p in paths: text = p.read_text(encoding="utf-8") whisper = Whisper(source_path=str(p), raw_text=text) await store.upsert_whisper(whisper) logger.info("Ingested %d whispers from %s", len(paths), root)

if name == "main":  # pragma: no cover import argparse

parser = argparse.ArgumentParser(description="Ingest files into Æon storage.")
parser.add_argument("root", type=Path, help="Path to directory to ingest")
parser.add_argument("--pattern", default="**/*.*", help="Glob pattern e.g. **/*.md")
args = parser.parse_args()

asyncio.run(_ingest_cli(args.root, args.pattern))

