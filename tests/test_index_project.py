from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

import main
from app.models import EmbeddedChunk


@pytest.fixture
def api(monkeypatch):
    llm = Mock()
    llm.generate_tests.return_value = "generated tests"
    monkeypatch.setitem(main.app.dependency_overrides, main.get_llm, lambda: llm)
    monkeypatch.setattr(main, "embed_chunks", lambda chunks: [
        EmbeddedChunk(**chunk.model_dump(), embedding=[1.0, 0.0], embedding_model="test")
        for chunk in chunks
    ])
    monkeypatch.setattr(main, "save_project", Mock())
    monkeypatch.setattr(main, "embed_query", Mock(return_value=[1.0] + [0.0] * 1023))
    monkeypatch.setattr(main, "search_chunks", Mock(return_value=[]))
    with TestClient(main.app) as client:
        yield client, llm


def test_indexing_and_health_without_llm(api, monkeypatch):
    client, llm = api
    monkeypatch.setattr(main, "LLMClient", Mock(side_effect=AssertionError("No LLM during indexing")))
    assert client.get("/health").json() == {"status": "ok"}
    response = client.post("/index-project", json={
        "project_name": "demo", "project_id": "project-one",
        "project_files": [
            {"path": "demo.py", "content": "def twice(x): return 2 * x"},
            {"path": "package.json", "content": '{"scripts":{"test":"jest"}}'},
        ],
    })
    assert response.status_code == 200
    result = response.json()
    assert result["files_received"] == result["chunks_created"] == len(result["chunks"]) == 2
    assert result["chunks"][0]["symbol_name"] == "twice"
    assert result["chunks"][1]["chunk_type"] == "config"
    assert result["chunks"][0]["embedding"] == [1.0, 0.0]
    assert main.save_project.call_args.args[0] == "project-one"
    empty = client.post("/index-project", json={"project_name": "empty", "project_files": []})
    assert empty.json() == {"project_name": "empty", "files_received": 0, "chunks_created": 0, "chunks": []}
    assert main.save_project.call_args.args == ("empty", [])
    main.LLMClient.assert_not_called()
    llm.generate_tests.assert_not_called()


@pytest.mark.parametrize("project_id,options,uses_context", [
    (None, {}, False),
    ("project-one", {}, True),
    ("project-one", {"use_rag": True}, True),
    ("project-one", {"use_rag": False}, False),
])
def test_selected_code_with_and_without_rag(api, project_id, options, uses_context):
    client, llm = api
    code = "def f(): return 1"
    main.search_chunks.return_value = [
        {"file_path": "a.py", "start_line": 1, "end_line": 1, "content": code},
        {"file_path": "helpers.py", "start_line": 4, "end_line": 5, "content": "def helper(): return 42"},
    ]
    response = client.post("/generate-tests", json={
        "code": code, "language": "python", "source_file": "a.py",
        "framework": "pytest", "instruction": "edge cases", "project_id": project_id,
        **options,
    })
    assert response.status_code == 200
    assert response.json() == {"tests": "generated tests", "symbol_name": "f"}
    context = "helpers.py:4-5\ndef helper(): return 42" if uses_context else ""
    llm.generate_tests.assert_called_with(
        code=code, language="python", source_file="a.py", framework="pytest",
        instruction="edge cases", context=context,
    )
    if uses_context:
        main.search_chunks.assert_called_once_with(project_id, main.embed_query.return_value)
        main.embed_query.assert_called_once_with("edge cases\n" + code)
    else:
        main.embed_query.assert_not_called()
        main.search_chunks.assert_not_called()


def test_free_request_uses_source_metadata(api):
    client, llm = api
    main.search_chunks.return_value = [
        {"file_path": "requirements.txt", "chunk_type": "config", "start_line": 1, "end_line": 1, "content": "pytest"},
        {"file_path": "a.py", "chunk_type": "import", "start_line": 1, "end_line": 1, "content": "import math"},
        {"file_path": "func.py", "language": "python", "chunk_type": "function", "symbol_name": "factorial",
         "start_line": 5, "end_line": 7, "content": "def factorial(n): return n"},
    ]
    response = client.post("/generate-tests", json={
        "code": "", "language": "auto", "project_id": "demo", "instruction": "Тесты для factorial",
    })
    assert response.json() == {"tests": "generated tests", "source_file": "func.py",
                              "language": "python", "symbol_name": "factorial"}
    args = llm.generate_tests.call_args.kwargs
    assert (args["code"], args["source_file"], args["framework"]) == ("def factorial(n): return n", "func.py", None)
    assert args["context"] == "requirements.txt:1-1\npytest\n\na.py:1-1\nimport math"


def test_context_limit_keeps_whole_chunks(api):
    client, llm = api
    main.search_chunks.return_value = [
        {"file_path": "first.py", "start_line": 1, "end_line": 1, "content": "x = 1"},
        {"file_path": "huge.py", "start_line": 1, "end_line": 1, "content": "x" * 8000},
        {"file_path": "last.py", "start_line": 1, "end_line": 1, "content": "y = 2"},
    ]
    response = client.post("/generate-tests", json={
        "code": "def f(): return 1", "language": "python", "project_id": "demo",
    })
    assert response.status_code == 200
    assert llm.generate_tests.call_args.kwargs["context"] == "first.py:1-1\nx = 1"


def test_generation_errors(api):
    client, llm = api
    response = client.post("/generate-tests", json={
        "code": " ", "language": "auto", "project_id": "demo",
        "instruction": "factorial", "use_rag": False,
    })
    assert response.status_code == 400
    assert response.json()["detail"] == "Для генерации без RAG передайте исходный код"
    main.embed_query.assert_not_called()
    main.search_chunks.assert_not_called()
    request = {"code": "", "language": "auto"}
    assert client.post("/generate-tests", json=request).status_code == 400
    request.update(project_id="demo", instruction="factorial")
    assert client.post("/generate-tests", json=request).status_code == 404
    llm.generate_tests.assert_not_called()
    llm.generate_tests.side_effect = RuntimeError("Provider unavailable")
    response = client.post("/generate-tests", json={"code": "def f(): return 1", "language": "python"})
    assert response.status_code == 500
    assert response.json()["detail"] == "Ошибка при генерации тестов"
