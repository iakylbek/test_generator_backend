from functools import lru_cache
from pathlib import Path

from .models import Chunk, EmbeddedChunk

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        MODEL_NAME,
        device="cpu",
        cache_folder=str(Path(__file__).resolve().parents[1] / ".model-cache"),
    )


def embed_chunks(chunks: list[Chunk]) -> list[EmbeddedChunk]:
    if not chunks:
        return []

    model = _get_model()
    texts = [chunk.content for chunk in chunks]
    lengths = model.tokenizer(texts, truncation=False, return_length=True)["length"]
    for chunk, length in zip(chunks, lengths, strict=True):
        if length > model.max_seq_length:
            raise ValueError(f"Chunk {chunk.file_path}:{chunk.start_line} exceeds the model token limit")

    vectors = model.encode(
        texts,
        batch_size=4,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [
        EmbeddedChunk(**chunk.model_dump(), embedding=vector.tolist(), embedding_model=MODEL_NAME)
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]


def embed_query(text: str) -> list[float]:
    vector = _get_model().encode(
        [text], prompt_name="query", normalize_embeddings=True, show_progress_bar=False,
    )[0]
    return vector.tolist()
