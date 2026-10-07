from functools import lru_cache
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from .models import EmbeddedChunk

COLLECTION = "project_chunks"
VECTOR_SIZE = 1024


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    path = Path(__file__).resolve().parents[1] / ".qdrant"
    return QdrantClient(path=str(path), force_disable_check_same_thread=True)


def save_project(project_id: str, chunks: list[EmbeddedChunk]) -> None:
    client = get_client()
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
        )

    points = [
        models.PointStruct(
            id=str(uuid5(NAMESPACE_URL, f"{project_id}:{chunk.file_path}:{chunk.start_byte}:{chunk.end_byte}")),
            vector=chunk.embedding,
            payload={"project_id": project_id, **chunk.model_dump(exclude={"embedding"})},
        )
        for chunk in chunks
    ]
    if points:
        client.upsert(collection_name=COLLECTION, points=points, wait=True)

    client.delete(
        collection_name=COLLECTION,
        points_selector=models.FilterSelector(filter=models.Filter(
            must=[models.FieldCondition(key="project_id", match=models.MatchValue(value=project_id))],
            must_not=[models.HasIdCondition(has_id=[point.id for point in points])] if points else None,
        )),
        wait=True,
    )


def search_chunks(project_id: str, vector: list[float], limit: int = 5) -> list[dict]:
    client = get_client()
    if not client.collection_exists(COLLECTION):
        return []
    result = client.query_points(
        collection_name=COLLECTION,
        query=vector,
        query_filter=models.Filter(must=[
            models.FieldCondition(key="project_id", match=models.MatchValue(value=project_id))
        ]),
        limit=limit,
        with_payload=True,
    )
    return [point.payload for point in result.points]
