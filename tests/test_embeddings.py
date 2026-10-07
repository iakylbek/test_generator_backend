from unittest.mock import Mock

import pytest

from app import embeddings
from app.chunker import ProjectChunker
from app.models import ProjectFile


def test_empty_input_does_not_load_model(monkeypatch):
    load_model = Mock(side_effect=AssertionError("Model must not be loaded"))
    monkeypatch.setattr(embeddings, "_get_model", load_model)
    assert embeddings.embed_chunks([]) == []
    load_model.assert_not_called()


def test_vectors_keep_chunk_order_and_metadata(monkeypatch):
    chunks = ProjectChunker().chunk_file(ProjectFile(
        path="example.py", content="def first(): return 1\n\ndef second(): return 2",
    ))
    before = [chunk.model_dump() for chunk in chunks]
    model = Mock(max_seq_length=32768)
    model.tokenizer.return_value = {"length": [8, 8]}
    model.encode.return_value = [Mock(), Mock()]
    model.encode.return_value[0].tolist.return_value = [1.0, 0.0]
    model.encode.return_value[1].tolist.return_value = [0.0, 1.0]
    monkeypatch.setattr(embeddings, "_get_model", lambda: model)

    result = embeddings.embed_chunks(chunks)

    assert [c.embedding for c in result] == [[1.0, 0.0], [0.0, 1.0]]
    assert all(c.embedding_model == embeddings.MODEL_NAME for c in result)
    assert [c.model_dump(exclude={"embedding", "embedding_model"}) for c in result] == before
    assert [c.model_dump() for c in chunks] == before
    model.encode.assert_called_once_with(
        [c.content for c in chunks], batch_size=4,
        normalize_embeddings=True, show_progress_bar=False,
    )


def test_long_chunk_is_rejected_instead_of_silently_truncated(monkeypatch):
    chunks = ProjectChunker().chunk_file(ProjectFile(path="a.py", content="def f(): return 1"))
    model = Mock(max_seq_length=10)
    model.tokenizer.return_value = {"length": [11]}
    monkeypatch.setattr(embeddings, "_get_model", lambda: model)
    with pytest.raises(ValueError, match="a.py:1"):
        embeddings.embed_chunks(chunks)
    model.encode.assert_not_called()
