"""FastAPI entrypoint implementing Aeon retrieval endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from pymongo import ReturnDocument
from pymongo.database import Database

from aeon_core import MongoManager, Neo4jManager, settings
from bson import ObjectId

from .schemas import (
    ComparativeView,
    GeoResponse,
    GeoResult,
    GraphEdge,
    GraphNode,
    GraphResponse,
    GlyphCatalog,
    SearchFilters,
    SearchResponse,
    TimelinePoint,
    TimelineResponse,
    WhisperIn,
    WhisperOut,
)
from .search import hybrid_search
from .glyphs import GLYPH_PHASES

app = FastAPI(title="Aeon Retrieval API", version="0.1.0")

_mongo_manager = MongoManager()
_neo4j_manager = Neo4jManager()


def _load_whisper(db: Database, whisper_id: str):
    doc = db["whispers"].find_one({"_id": whisper_id})
    if doc:
        return doc
    try:
        object_id = ObjectId(whisper_id)
    except Exception:  # ObjectId invalid
        return None
    return db["whispers"].find_one({"_id": object_id})


def get_db() -> Database:
    return _mongo_manager.db


def get_driver():
    return _neo4j_manager.driver


@app.on_event("shutdown")
def shutdown_event() -> None:
    _mongo_manager.close()
    _neo4j_manager.close()


@app.post("/whispers", response_model=Dict[str, str])
def upsert_whisper(payload: WhisperIn, db: Database = Depends(get_db)) -> Dict[str, str]:
    doc = payload.dict(exclude_none=True)
    doc.setdefault("created_at", datetime.utcnow())
    doc.setdefault("visibility", settings.default_visibility)

    lookup: Dict[str, object] = {}
    if doc.get("checksum"):
        lookup["checksum"] = doc["checksum"]
    else:
        lookup = {"title": doc["title"], "created_at": doc["created_at"]}

    result = db["whispers"].find_one_and_update(
        lookup,
        {"$set": doc},
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    if not result:
        raise HTTPException(status_code=500, detail="Failed to upsert whisper")
    return {"id": str(result["_id"])}


@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., description="User query"),
    visibility: Optional[str] = Query(None),
    chroma: Optional[List[str]] = Query(None),
    glyphs: Optional[List[str]] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    maat_min: Optional[float] = Query(None),
    bbox: Optional[List[float]] = Query(None),
    db: Database = Depends(get_db),
    driver=Depends(get_driver),
) -> SearchResponse:
    filters = SearchFilters(
        visibility=visibility,
        chroma=chroma,
        glyphs=glyphs,
        start_date=start_date,
        end_date=end_date,
        maat_min=maat_min,
        bbox=bbox,
    )
    results, explanations = hybrid_search(db, driver, q, filters)
    return SearchResponse(query=q, results=results, explanations=explanations)


@app.get("/map/{whisper_id}", response_model=GraphResponse)
def whisper_map(whisper_id: str, driver=Depends(get_driver)) -> GraphResponse:
    query = """
    MATCH (w:Whisper {id:$id})-[r:MENTIONS|USES_GLYPH|HAS_CHROMA|CITES|FOLLOWS*1..2]-(n)
    WITH collect(DISTINCT w) + collect(DISTINCT n) AS nodes
    UNWIND nodes AS n2
    WITH DISTINCT n2
    OPTIONAL MATCH (n2)-[e]-(m)
    WHERE m IN nodes
    RETURN
      collect(DISTINCT {id:id(n2), labels:labels(n2), props:properties(n2)}) AS nodes,
      collect(DISTINCT {source:id(startNode(e)), target:id(endNode(e)), type:type(e)}) AS edges
    """
    with driver.session() as session:
        record = session.run(query, id=whisper_id).single()
    if not record:
        return GraphResponse(nodes=[], edges=[])
    nodes = [GraphNode(id=str(n["id"]), labels=n["labels"], props=n["props"]) for n in record["nodes"]]
    edges = [GraphEdge(source=str(e["source"]), target=str(e["target"]), type=e["type"]) for e in record["edges"]]
    return GraphResponse(nodes=nodes, edges=edges)


@app.get("/compare", response_model=ComparativeView)
def compare(theme: str = Query(..., description="Concept or entity to contrast"), db: Database = Depends(get_db), driver=Depends(get_driver)) -> ComparativeView:
    query = """
    MATCH (c:Entity {name:$theme})<-[:MENTIONS]-(w:Whisper)
    OPTIONAL MATCH (w)-[r:ALIGNS_WITH]->(c)
    RETURN w.id AS id, coalesce(r.score, 0) AS score
    """
    supportive: List[WhisperOut] = []
    critical: List[WhisperOut] = []
    with driver.session() as session:
        for record in session.run(query, theme=theme):
            whisper_id = record["id"]
            score = record["score"]
            if not whisper_id:
                continue
            doc = _load_whisper(db, whisper_id)
            if not doc:
                continue
            doc["_id"] = str(doc["_id"])
            doc["score"] = float(score)
            whisper = WhisperOut.parse_obj(doc)
            if score < 0:
                critical.append(whisper)
            else:
                supportive.append(whisper)
    supportive = sorted(supportive, key=lambda x: x.score or 0, reverse=True)[:10]
    critical = sorted(critical, key=lambda x: x.score or 0)[:10]
    return ComparativeView(supportive=supportive, critical=critical)


@app.get("/timeline", response_model=TimelineResponse)
def timeline(entity: str = Query(...), db: Database = Depends(get_db)) -> TimelineResponse:
    pipeline = [
        {"$match": {"entities.ref": entity}},
        {
            "$group": {
                "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$created_at"}},
                "count": {"$sum": 1},
            }
        },
        {"$sort": {"_id": 1}},
    ]
    points: List[TimelinePoint] = []
    for row in db["whispers"].aggregate(pipeline):
        day_str = row["_id"]
        day = datetime.strptime(day_str, "%Y-%m-%d")
        points.append(TimelinePoint(day=day, count=row["count"]))
    return TimelineResponse(entity=entity, points=points)


@app.get("/geo", response_model=GeoResponse)
def geo(
    bbox: List[float] = Query(..., description="Bounding box [minLon, minLat, maxLon, maxLat]"),
    db: Database = Depends(get_db),
) -> GeoResponse:
    if len(bbox) != 4:
        raise HTTPException(status_code=400, detail="Bounding box must contain four coordinates")
    min_lon, min_lat, max_lon, max_lat = bbox
    criteria = {
        "geo.lat": {"$gte": min_lat, "$lte": max_lat},
        "geo.lon": {"$gte": min_lon, "$lte": max_lon},
    }
    cursor = db["whispers"].find(criteria)
    results: List[GeoResult] = []
    for doc in cursor:
        geo = doc.get("geo") or {}
        results.append(
            GeoResult(
                id=str(doc["_id"]),
                title=doc.get("title", "Untitled"),
                lat=geo.get("lat"),
                lon=geo.get("lon"),
                visibility=doc.get("visibility", settings.default_visibility),
                score=(doc.get("ethics") or {}).get("maat_score"),
            )
        )
    return GeoResponse(results=results)




@app.get("/glyphs", response_model=GlyphCatalog)
def glyph_catalog() -> GlyphCatalog:
    """Return the full hyperglyph phase catalog including new octave entries."""
    return GlyphCatalog(phases=GLYPH_PHASES)

@app.get("/health", tags=["meta"])
def health() -> Dict[str, str]:
    return {
        "status": "ok",
        "mongo_uri": settings.mongo_uri,
        "neo4j_uri": settings.neo4j_uri,
    }
