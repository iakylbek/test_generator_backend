from functools import lru_cache
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from tree_sitter_language_pack import PackConfig, configure

from app.chunker import ProjectChunker
from app.embeddings import embed_chunks, embed_query
from app.llm import LLMClient
from app.vector_store import save_project, search_chunks
from app.models import (
    ProjectFile,
    ProjectIndexRequest,
    ProjectIndexResponse,
    TestGenerationRequest,
    TestGenerationResponse,
)

load_dotenv()
configure(PackConfig(cache_dir=os.getenv(
    "TREE_SITTER_CACHE_DIR", str(Path(__file__).parent / ".tree-sitter-cache")
)))

app = FastAPI(title="Unit Test Generator API", version="1.0.0")
chunker = ProjectChunker()


@lru_cache
def get_llm() -> LLMClient:
    return LLMClient()


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/index-project", response_model=ProjectIndexResponse)
def index_project(request: ProjectIndexRequest):
    chunks = embed_chunks(chunker.chunk_project(request.project_files))
    save_project(request.project_id or request.project_name, chunks)
    return ProjectIndexResponse(
        project_name=request.project_name,
        files_received=len(request.project_files),
        chunks_created=len(chunks),
        chunks=chunks,
    )


@app.post("/generate-tests", response_model=TestGenerationResponse, response_model_exclude_none=True)
def generate_tests(request: TestGenerationRequest, llm: LLMClient = Depends(get_llm)):
    from_request = not request.code.strip()
    if from_request and not request.use_rag:
        raise HTTPException(status_code=400, detail="Для генерации без RAG передайте исходный код")
    if from_request and (not request.project_id or not request.instruction):
        raise HTTPException(status_code=400, detail="Укажите запрос и project_id для поиска кода")
    try:
        code = request.code
        language = request.language
        source_file = request.source_file
        symbol_name = None
        context = ""
        if request.use_rag and request.project_id:
            query = request.instruction if from_request else f"{request.instruction or ''}\n{code}"
            matches = search_chunks(request.project_id, embed_query(query))
            if from_request:
                target = next((item for item in matches if item.get("chunk_type") not in ("import", "config")
                               and item.get("content", "").strip()), None)
                if target is None:
                    raise HTTPException(status_code=404, detail="В индексе проекта не найден код для тестирования")
                code = target["content"]
                language = target["language"]
                source_file = target["file_path"]
                symbol_name = target.get("symbol_name")
            for item in matches:
                if item.get("content", "").strip() == code.strip():
                    continue
                part = f"{item['file_path']}:{item['start_line']}-{item['end_line']}\n{item['content']}"
                addition = ("\n\n" if context else "") + part
                if len(context) + len(addition) > 8000:
                    break
                context += addition
        if not symbol_name:
            chunks = chunker.chunk_file(ProjectFile(path=source_file or "snippet", content=code))
            symbol_name = next((item.symbol_name for item in chunks
                                if item.chunk_type in ("function", "method") and item.symbol_name), None)
        tests = llm.generate_tests(
            code=code,
            language=language,
            source_file=source_file,
            framework=request.framework,
            instruction=request.instruction,
            context=context,
        )
        return TestGenerationResponse(
            tests=tests, symbol_name=symbol_name,
            source_file=source_file if from_request else None,
            language=language if from_request else None,
        )

    except HTTPException:
        raise
    except Exception as error:
        print(error)
        raise HTTPException(status_code=500, detail="Ошибка при генерации тестов") from error
