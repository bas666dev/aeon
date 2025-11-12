"""Database connection helpers."""

from contextlib import contextmanager
from typing import Generator, Optional

from neo4j import GraphDatabase, Driver
from pymongo import MongoClient

from .config import settings


class MongoManager:
    """Manage lifecycle of the MongoDB client."""

    def __init__(self, uri: Optional[str] = None, *, db_name: Optional[str] = None) -> None:
        self._uri = uri or settings.mongo_uri
        self._db_name = db_name or settings.mongo_db
        self._client: Optional[MongoClient] = None

    @property
    def client(self) -> MongoClient:
        if self._client is None:
            self._client = MongoClient(self._uri)
        return self._client

    @property
    def db(self):  # type: ignore[override]
        return self.client[self._db_name]

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


class Neo4jManager:
    """Manage lifecycle of the Neo4j driver."""

    def __init__(self, uri: Optional[str] = None, *, user: Optional[str] = None, password: Optional[str] = None) -> None:
        self._uri = uri or settings.neo4j_uri
        self._user = user or settings.neo4j_user
        self._password = password or settings.neo4j_password
        self._driver: Optional[Driver] = None

    @property
    def driver(self) -> Driver:
        if self._driver is None:
            self._driver = GraphDatabase.driver(self._uri, auth=(self._user, self._password))
        return self._driver

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None


@contextmanager
def mongo_session(manager: MongoManager) -> Generator:
    try:
        yield manager.db
    finally:
        manager.close()


@contextmanager
def neo4j_session(manager: Neo4jManager) -> Generator:
    session = manager.driver.session()
    try:
        yield session
    finally:
        session.close()
        manager.close()


__all__ = ["MongoManager", "Neo4jManager", "mongo_session", "neo4j_session"]
