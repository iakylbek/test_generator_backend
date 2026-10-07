from math import sqrt
from time import perf_counter

from app.chunker import ProjectChunker
from app.embeddings import embed_chunks
from app.models import ProjectFile


source = ProjectFile(path="example.py", content="def twice(x):\n    return x * 2\n")
chunks = ProjectChunker().chunk_file(source)
started = perf_counter()
result = embed_chunks(chunks)

assert len(result) == 1
assert result[0].file_path == source.path
assert result[0].symbol_name == "twice"
assert len(result[0].embedding) == 1024
assert abs(sqrt(sum(value * value for value in result[0].embedding)) - 1) < 0.001

print(f"Модель: {result[0].embedding_model}")
print(f"Символ: {result[0].symbol_name}")
print(f"Размер вектора: {len(result[0].embedding)}")
print(f"Время: {perf_counter() - started:.2f} с")
