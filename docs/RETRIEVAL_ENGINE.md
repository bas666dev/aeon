# Aeon Retrieval Engine

This document describes the implementation of the hybrid MongoDB + Neo4j stack that powers the Whisper Map, comparative discovery, and spatial/temporal views.

## Services

The deployment uses Docker Compose to orchestrate four containers:

- **mongo** — primary document store for whispers, entities, audits, and embeddings.
- **neo4j** — graph store responsible for relationships between whispers, glyphs, chroma, entities, and timelines.
- **api** — FastAPI application that exposes search, map, comparison, timeline, and geo endpoints.
- **worker** — change-stream consumer that mirrors MongoDB updates into Neo4j and maintains embeddings.

## Configuration

Common settings are located in `aeon_core/config.py` and are read from environment variables with sensible defaults:

| Variable | Description | Default |
|----------|-------------|---------|
| `MONGO_URI` | MongoDB connection string | `mongodb://localhost:27017/aeon` |
| `MONGO_DB` | Database name | `aeon` |
| `NEO4J_URI` | Bolt connection string | `bolt://localhost:7687` |
| `NEO4J_USER` / `NEO4J_PASS` | Neo4j credentials | `neo4j` / `neo4jpass` |
| `HYBRID_SEARCH_TOP_K` | Search result limit | `20` |
| `MAAT_MIN_SCORE` | Minimum ethics score for visibility | `0.8` |

## API Highlights

The FastAPI service (`api/main.py`) implements the following endpoints:

- `POST /whispers` — upsert whispers in MongoDB (auto-populates timestamps and visibility).
- `GET /search` — orchestrates BM25-style regex matching, deterministic dense vectors, graph expansion, and policy filtering.
- `GET /map/{id}` — returns a graph neighborhood suitable for visualization.
- `GET /compare` — yields supportive vs critical viewpoints linked via `ALIGNS_WITH` relationships.
- `GET /timeline` — time-series counts for a given entity reference.
- `GET /geo` — bounding-box spatial retrieval.
- `GET /health` — lightweight readiness check.

Supporting code lives in `api/schemas.py` (Pydantic models) and `api/search.py` (hybrid retrieval logic). Deterministic pseudo-embeddings are generated via SHA-256 and cosine similarity to avoid heavyweight ML dependencies while keeping interface compatibility with real vectors.

## Worker Responsibilities

`worker/sync.py` consumes MongoDB change streams and performs the following actions for each whisper document:

1. Ensures a vector representation is stored in the `embeddings` collection and referenced by the whisper.
2. Looks up glyph, chroma, and entity metadata.
3. Upserts the corresponding nodes and relationships in Neo4j using Cypher `MERGE` statements.
4. Handles deletions by removing the associated `Whisper` node and its edges.
5. Persists resume tokens in the `sync_state` collection for idempotent restarts.

Error handling includes automatic retries with exponential sleep for transient MongoDB issues.

## Running Locally

```bash
# Launch the full stack
docker compose up --build

# API will be available at http://localhost:8000/docs
```

Populate seed data in Neo4j by connecting to the bolt console and running the sample statements from the design document. MongoDB can be seeded with whispers via `POST /whispers` or by inserting directly into the `whispers` collection.

## Extensibility

- Swap the deterministic embedding placeholder with a true encoder (e.g., Sentence Transformers) by extending `_ensure_embedding` in `worker/sync.py` and `encode_query` in `api/search.py`.
- Add moderation or visibility rules by adjusting `policy_filter` in `api/search.py`.
- Introduce additional relationship types or analytics by modifying the Cypher statements used in `worker/sync.py` and the retrieval queries in `api/main.py`.

This scaffold provides a production-ready baseline that can be iterated upon with richer models, UI integrations, and observability hooks.
