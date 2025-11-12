"""MongoDB change stream worker that projects whispers into Neo4j."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Dict, Iterable, List, Optional

from neo4j import Driver
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import PyMongoError

from aeon_core import MongoManager, Neo4jManager, settings
from api.search import encode_query

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _lookup_entity(db: Database, ref: str) -> Dict[str, str]:
    entity = db["entities"].find_one({"_id": ref}) or db["entities"].find_one({"ref": ref})
    if not entity:
        return {"id": ref, "name": ref, "type": "Concept"}
    return {
        "id": str(entity.get("_id", ref)),
        "name": entity.get("name", ref),
        "type": entity.get("type", "Concept"),
    }


def _lookup_chroma(db: Database, names: Iterable[str]) -> List[Dict[str, Optional[str]]]:
    chroma_collection: Collection = db["chroma"]
    payload: List[Dict[str, Optional[str]]] = []
    for name in names:
        entry = chroma_collection.find_one({"name": name})
        payload.append({"name": name, "hex": entry.get("hex") if entry else None})
    return payload


def _ensure_embedding(db: Database, doc: Dict) -> List[str]:
    vector_ids = doc.get("vector_ids", [])
    if vector_ids:
        return vector_ids
    vector_id = f"vec_{doc['_id']}_body"
    vector = encode_query(doc.get("body", ""))
    db["embeddings"].replace_one(
        {"_id": vector_id},
        {"_id": vector_id, "values": vector.tolist(), "item_id": str(doc["_id"]), "kind": "whisper_body"},
        upsert=True,
    )
    db["whispers"].update_one({"_id": doc["_id"]}, {"$addToSet": {"vector_ids": vector_id}})
    return [vector_id]


def _format_datetime(value: Optional[datetime]) -> Optional[str]:
    if not value:
        return None
    return value.isoformat()


def upsert_graph(driver: Driver, db: Database, doc: Dict) -> None:
    whisper_id = str(doc["_id"])
    glyphs = doc.get("glyphs", [])
    chroma_payload = _lookup_chroma(db, doc.get("chroma", []))
    entity_payload = []
    for entity in doc.get("entities", []):
        ref = entity.get("ref")
        if not ref:
            continue
        info = _lookup_entity(db, ref)
        info["role"] = entity.get("role")
        entity_payload.append(info)

    vector_ids = _ensure_embedding(db, doc)

    cypher = """
    MERGE (w:Whisper {id:$id})
    SET w.title=$title,
        w.created_at=$created_at,
        w.visibility=$visibility,
        w.language=$language,
        w.maat_score=$maat_score
    WITH w
    FOREACH (glyph IN $glyphs |
        MERGE (g:Glyph {symbol:glyph})
        MERGE (w)-[:USES_GLYPH]->(g)
    )
    WITH w
    FOREACH (chroma IN $chroma |
        MERGE (c:Chroma {name:chroma.name})
        SET c.hex = chroma.hex
        MERGE (w)-[:HAS_CHROMA]->(c)
    )
    WITH w
    FOREACH (entity IN $entities |
        MERGE (e:Entity {id:entity.id})
        SET e.name = entity.name,
            e.type = entity.type
        MERGE (w)-[rel:MENTIONS]->(e)
        SET rel.role = entity.role
    )
    """
    payload = {
        "id": whisper_id,
        "title": doc.get("title"),
        "created_at": _format_datetime(doc.get("created_at")),
        "visibility": doc.get("visibility", settings.default_visibility),
        "language": doc.get("language"),
        "maat_score": (doc.get("ethics") or {}).get("maat_score"),
        "glyphs": glyphs,
        "chroma": chroma_payload,
        "entities": entity_payload,
        "vector_ids": vector_ids,
    }
    with driver.session() as session:
        session.run(cypher, **payload)


def remove_from_graph(driver: Driver, whisper_id: str) -> None:
    cypher = "MATCH (w:Whisper {id:$id}) DETACH DELETE w"
    with driver.session() as session:
        session.run(cypher, id=whisper_id)


class ChangeStreamWorker:
    """Stream MongoDB changes and mirror them into Neo4j."""

    def __init__(self, mongo_manager: Optional[MongoManager] = None, neo4j_manager: Optional[Neo4jManager] = None) -> None:
        self.mongo = mongo_manager or MongoManager()
        self.neo4j = neo4j_manager or Neo4jManager()
        self.db = self.mongo.db
        self.driver = self.neo4j.driver
        self.sync_state = self.db["sync_state"]

    def _load_resume_token(self):
        state = self.sync_state.find_one({"_id": "whispers"})
        return state.get("resume_token") if state else None

    def _store_resume_token(self, token) -> None:
        self.sync_state.update_one(
            {"_id": "whispers"},
            {"$set": {"resume_token": token, "updated_at": datetime.utcnow()}},
            upsert=True,
        )

    def run(self) -> None:
        resume_token = self._load_resume_token()
        logger.info("Starting change stream with resume token: %s", resume_token)
        while True:
            watch_kwargs = {"full_document": "updateLookup"}
            if resume_token:
                watch_kwargs["resume_after"] = resume_token
            try:
                with self.db["whispers"].watch(**watch_kwargs) as stream:
                    for change in stream:
                        resume_token = change.get("_id") or resume_token
                        operation = change.get("operationType")
                        if operation == "delete":
                            key = change.get("documentKey", {}).get("_id")
                            if key:
                                remove_from_graph(self.driver, str(key))
                        else:
                            doc = change.get("fullDocument")
                            if doc:
                                upsert_graph(self.driver, self.db, doc)
                        if resume_token:
                            self._store_resume_token(resume_token)
            except PyMongoError as exc:
                logger.exception("Change stream error: %s", exc)
                time.sleep(2)
                continue
            except KeyboardInterrupt:
                logger.info("Worker interrupted, shutting down")
                break

    def close(self) -> None:
        self.mongo.close()
        self.neo4j.close()


def main() -> None:
    worker = ChangeStreamWorker()
    try:
        worker.run()
    finally:
        worker.close()


if __name__ == "__main__":
    main()
