"""Pydantic models shared by API endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class EntityRef(BaseModel):
    ref: str
    role: Optional[str] = None


class SourceRef(BaseModel):
    ref: str


class Timeline(BaseModel):
    started: Optional[datetime] = None
    ended: Optional[datetime] = None


class GeoPoint(BaseModel):
    lat: float
    lon: float
    label: Optional[str] = None


class EthicsEnvelope(BaseModel):
    maat_score: float = Field(..., ge=0, le=1)
    flags: List[str] = Field(default_factory=list)


class WhisperIn(BaseModel):
    title: str
    body: str
    language: Optional[str] = None
    created_at: Optional[datetime] = None
    tags: List[str] = Field(default_factory=list)
    glyphs: List[str] = Field(default_factory=list)
    chroma: List[str] = Field(default_factory=list)
    entities: List[EntityRef] = Field(default_factory=list)
    source: Optional[SourceRef] = None
    timeline: Optional[Timeline] = None
    geo: Optional[GeoPoint] = None
    visibility: str = Field(default="private")
    ethics: Optional[EthicsEnvelope] = None
    vector_ids: List[str] = Field(default_factory=list)
    checksum: Optional[str] = None

    @validator("visibility")
    def validate_visibility(cls, value: str) -> str:
        allowed = {"private", "shared", "public"}
        if value not in allowed:
            raise ValueError(f"visibility must be one of {allowed}")
        return value


class WhisperOut(BaseModel):
    id: str = Field(..., alias="_id")
    title: str
    body: str
    language: Optional[str]
    created_at: datetime
    tags: List[str] = Field(default_factory=list)
    glyphs: List[str] = Field(default_factory=list)
    chroma: List[str] = Field(default_factory=list)
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    visibility: str
    ethics: Optional[EthicsEnvelope] = None
    score: Optional[float] = None

    class Config:
        allow_population_by_field_name = True


class SearchFilters(BaseModel):
    visibility: Optional[str] = None
    chroma: Optional[List[str]] = None
    glyphs: Optional[List[str]] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    maat_min: Optional[float] = None
    bbox: Optional[List[float]] = None


class SearchResponse(BaseModel):
    query: str
    results: List[WhisperOut]
    explanations: List[str]


class GraphNode(BaseModel):
    id: str
    labels: List[str]
    props: Dict[str, Any]


class GraphEdge(BaseModel):
    source: str
    target: str
    type: str


class GraphResponse(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]


class ComparativeView(BaseModel):
    supportive: List[WhisperOut] = Field(default_factory=list)
    critical: List[WhisperOut] = Field(default_factory=list)


class TimelinePoint(BaseModel):
    day: datetime
    count: int


class TimelineResponse(BaseModel):
    entity: str
    points: List[TimelinePoint]


class GeoResult(BaseModel):
    id: str
    title: str
    lat: float
    lon: float
    visibility: str
    score: Optional[float] = None


class GeoResponse(BaseModel):
    results: List[GeoResult]


class GlyphPhase(BaseModel):
    phase: int
    symbol: str
    name: str
    function: str
    appearance: str
    causal_drift: str
    tone: Optional[str] = None
    usage: str
    status: Optional[str] = None
    notes: Optional[str] = None
