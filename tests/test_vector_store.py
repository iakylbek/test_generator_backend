from qdrant_client import QdrantClient

from app import vector_store
from app.models import EmbeddedChunk


def make_chunk(path: str, content: str) -> EmbeddedChunk:
    return EmbeddedChunk(
        file_path=path, language="python", chunk_type="function",
        symbol_name="example", start_line=1, end_line=1,
        content=content, start_byte=0, end_byte=len(content.encode()),
        parser="tree-sitter", embedding=[1.0] + [0.0] * 1023,
        embedding_model="test-model",
    )


def test_project_snapshot_persists_and_replaces_only_its_own_points(tmp_path, monkeypatch):
    client = QdrantClient(path=str(tmp_path))
    monkeypatch.setattr(vector_store, "get_client", lambda: client)

    vector_store.save_project("one", [make_chunk("a.py", "old"), make_chunk("b.py", "b")])
    vector_store.save_project("two", [make_chunk("a.py", "other")])
    vector_store.save_project("one", [make_chunk("a.py", "new")])
    vector_store.save_project("one", [make_chunk("a.py", "new")])
    client.close()

    reopened = QdrantClient(path=str(tmp_path))
    try:
        monkeypatch.setattr(vector_store, "get_client", lambda: reopened)
        records, _ = reopened.scroll(vector_store.COLLECTION, limit=10, with_vectors=True)
        assert len(records) == 2
        assert {(r.payload["project_id"], r.payload["content"]) for r in records} == {
            ("one", "new"), ("two", "other"),
        }
        assert all(r.payload["file_path"] == "a.py" for r in records)
        assert all(r.payload["embedding_model"] == "test-model" for r in records)
        assert all(len(r.vector) == 1024 for r in records)

        matches = vector_store.search_chunks("one", [1.0] + [0.0] * 1023)
        assert [item["content"] for item in matches] == ["new"]
        assert vector_store.search_chunks("missing", [1.0] + [0.0] * 1023) == []

        vector_store.save_project("one", [])
        remaining, _ = reopened.scroll(vector_store.COLLECTION, limit=10)
        assert len(remaining) == 1
        assert remaining[0].payload["project_id"] == "two"
    finally:
        reopened.close()
