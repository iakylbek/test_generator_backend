from pydantic import BaseModel, Field


class ProjectFile(BaseModel):
    path: str
    content: str


class Chunk(BaseModel):

    file_path: str
    language: str
    chunk_type: str
    symbol_name: str | None = None
    parent_symbol: str | None = None
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    content: str
    start_byte: int = Field(ge=0)
    end_byte: int = Field(ge=0)
    parser: str


class EmbeddedChunk(Chunk):
    embedding: list[float]
    embedding_model: str


class ProjectIndexRequest(BaseModel):
    project_name: str
    project_id: str | None = None
    project_files: list[ProjectFile]


class ProjectIndexResponse(BaseModel):
    project_name: str
    files_received: int
    chunks_created: int
    chunks: list[EmbeddedChunk]


class TestGenerationRequest(BaseModel):
    code: str
    language: str
    use_rag: bool = True
    project_id: str | None = None
    source_file: str | None = None
    framework: str | None = None
    instruction: str | None = None


class TestGenerationResponse(BaseModel):
    tests: str
    source_file: str | None = None
    language: str | None = None
    symbol_name: str | None = None
