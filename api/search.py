"""Hybrid search routines combining MongoDB and Neo4j signals."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from bson import ObjectId
from neo4j import Driver
from pymongo.database import Database

from aeon_core.config import settings

from .schemas import SearchFilters, WhisperOut


_TOKEN_SPLIT = re.compile(r"[^\w𐐀-𐑏]+", re.UNICODE)


def _tokenize(query: str) -> List[str]:
    return [t for t in _TOKEN_SPLIT.split(query.lower()) if t]


def encode_query(query: str, dims: int = 64) -> np.ndarray:
    """Generate a deterministic pseudo-embedding for the query string."""

    digest = hashlib.sha256(query.encode("utf-8")).digest()
    raw = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
    if raw.size < dims:
        raw = np.tile(raw, int(math.ceil(dims / raw.size)))
    vector = raw[:dims]
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def _fetch_embeddings(db: Database, vector_ids: Sequence[str]) -> Dict[str, np.ndarray]:
    if not vector_ids:
        return {}
    docs = db["embeddings"].find({"_id": {"$in": list(vector_ids)}})
    vectors: Dict[str, np.ndarray] = {}
    for doc in docs:
        values = doc.get("values") or doc.get("vector") or []
        vectors[str(doc["_id"])] = np.asarray(values, dtype=np.float32)
    return vectors


def search_lexical(db: Database, query: str, filters: SearchFilters, limit: int = 50) -> List[Dict]:
    tokens = _tokenize(query)
    regex = re.compile("|".join(re.escape(t) for t in tokens), re.IGNORECASE) if tokens else None

    criteria: Dict[str, object] = {}
    if regex:
        criteria["$or"] = [
            {"title": {"$regex": regex}},
            {"body": {"$regex": regex}},
            {"tags": {"$in": tokens}},
        ]
    if filters.visibility:
        criteria["visibility"] = filters.visibility
    if filters.chroma:
        criteria["chroma"] = {"$in": filters.chroma}
    if filters.glyphs:
        criteria["glyphs"] = {"$in": filters.glyphs}
    if filters.start_date or filters.end_date:
        created_range: Dict[str, object] = {}
        if filters.start_date:
            created_range["$gte"] = filters.start_date
        if filters.end_date:
            created_range["$lte"] = filters.end_date
        criteria["created_at"] = created_range

    cursor = db["whispers"].find(criteria).limit(limit)
    results: List[Dict] = []
    for doc in cursor:
        doc = dict(doc)
        doc["_id"] = str(doc["_id"])
        doc["score"] = 0.5  # baseline score for lexical hits
        results.append(doc)
    return results


def search_vectors(db: Database, query: str, filters: SearchFilters, limit: int = 50) -> List[Dict]:
    query_vector = encode_query(query)
    criteria: Dict[str, object] = {}
    if filters.visibility:
        criteria["visibility"] = filters.visibility
    if filters.chroma:
        criteria["chroma"] = {"$in": filters.chroma}
    if filters.glyphs:
        criteria["glyphs"] = {"$in": filters.glyphs}

    cursor = db["whispers"].find(criteria, {"vector_ids": 1})
    scored: List[Tuple[str, float]] = []
    id_to_doc: Dict[str, Dict] = {}
    for doc in cursor:
        vector_ids = doc.get("vector_ids", [])
        embeddings = _fetch_embeddings(db, vector_ids)
        best = 0.0
        for vector_id, vector in embeddings.items():
            if vector.size == 0:
                continue
            norm = np.linalg.norm(vector)
            if norm == 0:
                continue
            score = float(np.dot(query_vector, vector / norm))
            best = max(best, score)
        if best > 0:
            _id = str(doc["_id"])
            scored.append((_id, best))
            id_to_doc[_id] = {"_id": _id, "score": best}
    scored.sort(key=lambda x: x[1], reverse=True)
    top_ids = [doc_id for doc_id, _ in scored[:limit]]
    object_ids = []
    string_ids = []
    for doc_id in top_ids:
        try:
            object_ids.append(ObjectId(doc_id))
        except Exception:
            string_ids.append(doc_id)
    criteria: Dict[str, object] = {}
    clauses = []
    if object_ids:
        clauses.append({"_id": {"$in": object_ids}})
    if string_ids:
        clauses.append({"_id": {"$in": string_ids}})
    if not clauses:
        return []
    if len(clauses) == 1:
        criteria.update(clauses[0])
    else:
        criteria["$or"] = clauses
    docs = db["whispers"].find(criteria)
    result_docs: Dict[str, Dict] = {}
    for doc in docs:
        doc = dict(doc)
        doc_id = str(doc["_id"])
        doc["_id"] = doc_id
        doc["score"] = id_to_doc.get(doc_id, {}).get("score", 0)
        result_docs[doc_id] = doc
    return list(result_docs.values())


def expand_via_graph(tokens: Iterable[str], driver: Driver) -> Dict[str, List[str]]:
    if not tokens:
        return {"entities": [], "glyphs": [], "chroma": []}

    expansions: Dict[str, set] = {"entities": set(), "glyphs": set(), "chroma": set()}
    query = """
    UNWIND $tokens AS token
    MATCH (n)
    WHERE (n:Entity AND toLower(n.name) = token)
       OR (n:Glyph AND toLower(n.symbol) = token)
       OR (n:Chroma AND toLower(n.name) = token)
    WITH DISTINCT n
    OPTIONAL MATCH (n)-[:MENTIONS|RELATES_TO|USES_GLYPH|HAS_CHROMA|ALIGNS_WITH|FOLLOWS]-(m)
    RETURN collect(DISTINCT {labels: labels(n), value: coalesce(n.name, n.symbol)}) AS primary,
           collect(DISTINCT {labels: labels(m), value: coalesce(m.name, m.symbol)}) AS neighbors
    """
    with driver.session() as session:
        record = session.run(query, tokens=[t.lower() for t in tokens]).single()
    if not record:
        return {k: [] for k in expansions}

    for item in record["primary"] or []:
        labels = item.get("labels") or []
        value = item.get("value")
        if not value:
            continue
        if "Entity" in labels:
            expansions["entities"].add(value)
        if "Glyph" in labels:
            expansions["glyphs"].add(value)
        if "Chroma" in labels:
            expansions["chroma"].add(value)

    for item in record["neighbors"] or []:
        labels = item.get("labels") or []
        value = item.get("value")
        if not value:
            continue
        if "Entity" in labels:
            expansions["entities"].add(value)
        if "Glyph" in labels:
            expansions["glyphs"].add(value)
        if "Chroma" in labels:
            expansions["chroma"].add(value)

    return {k: sorted(v) for k, v in expansions.items()}


def _apply_graph_boost(result: Dict, expansions: Dict[str, List[str]]) -> None:
    score = result.get("score", 0.0)
    glyphs = {g.lower() for g in result.get("glyphs", [])}
    chroma = {c.lower() for c in result.get("chroma", [])}
    entities = {e.get("ref", "").lower() for e in result.get("entities", [])}

    if any(g.lower() in glyphs for g in expansions.get("glyphs", [])):
        score += 0.1
    if any(c.lower() in chroma for c in expansions.get("chroma", [])):
        score += 0.1
    if any(e.lower() in entities for e in expansions.get("entities", [])):
        score += 0.15
    result["score"] = score


def policy_filter(results: Iterable[Dict], filters: SearchFilters) -> List[Dict]:
    filtered: List[Dict] = []
    maat_min = filters.maat_min or settings.maat_min_score
    for doc in results:
        visibility = doc.get("visibility", settings.default_visibility)
        if filters.visibility and visibility != filters.visibility:
            continue
        ethics = doc.get("ethics", {}) or {}
        maat_score = ethics.get("maat_score", 1.0)
        if maat_score < maat_min:
            continue
        filtered.append(doc)
    return filtered


def cross_encode_rerank(results: List[Dict]) -> List[Dict]:
    return sorted(results, key=lambda doc: doc.get("score", 0.0), reverse=True)


def hybrid_search(db: Database, driver: Driver, query: str, filters: Optional[SearchFilters] = None) -> Tuple[List[WhisperOut], List[str]]:
    filters = filters or SearchFilters()
    tokens = _tokenize(query)
    expansions = expand_via_graph(tokens, driver)

    lexical_hits = search_lexical(db, query, filters)
    vector_hits = search_vectors(db, query, filters)

    merged: Dict[str, Dict] = {}
    for source, weight in ((lexical_hits, 0.6), (vector_hits, 0.9)):
        for doc in source:
            doc_id = doc["_id"]
            existing = merged.get(doc_id)
            score = doc.get("score", 0.0) * weight
            if existing:
                existing["score"] = max(existing.get("score", 0.0), score)
            else:
                merged[doc_id] = dict(doc)
                merged[doc_id]["score"] = score

    for doc in merged.values():
        _apply_graph_boost(doc, expansions)

    filtered = policy_filter(merged.values(), filters)
    reranked = cross_encode_rerank(filtered)
    limited = reranked[: settings.hybrid_search_top_k]

    results = [WhisperOut.parse_obj(doc) for doc in limited]
    explanations = []
    if expansions["entities"]:
        explanations.append("Entity emphasis: " + ", ".join(expansions["entities"]))
    if expansions["glyphs"]:
        explanations.append("Glyph resonance: " + ", ".join(expansions["glyphs"]))
    if expansions["chroma"]:
        explanations.append("Chroma alignment: " + ", ".join(expansions["chroma"]))

    return results, explanations
