"""Worker package that synchronises MongoDB change streams with Neo4j."""

from .sync import ChangeStreamWorker

__all__ = ["ChangeStreamWorker"]
