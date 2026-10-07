import pytest
from tree_sitter_language_pack import get_parser

from app.chunker import ChunkerConfig, ProjectChunker
from app.chunking_languages import COMMON_NODE_TYPES, LanguageProfile, LanguageRegistry
from app.models import ProjectFile


def chunks_for(path, content, limit=4000, overlap=3, **options):
    file = ProjectFile(path=path, content=content)
    chunks = ProjectChunker(ChunkerConfig(limit, overlap), **options).chunk_file(file)
    source = content.encode("utf-8")
    covered = bytearray(len(source))
    for chunk in chunks:
        assert chunk.file_path == path
        assert 0 < len(chunk.content) <= limit
        assert source[chunk.start_byte:chunk.end_byte].decode("utf-8") == chunk.content
        assert chunk.start_line == source[:chunk.start_byte].count(b"\n") + 1
        assert chunk.end_line == source[:chunk.end_byte - 1].count(b"\n") + 1
        covered[chunk.start_byte:chunk.end_byte] = b"\x01" * (chunk.end_byte - chunk.start_byte)
    assert all(covered[i] or chr(byte).isspace() for i, byte in enumerate(source))
    return chunks


def test_languages():
    examples = [
        ("a.py", "python", "import os\ndef twice(x): return x * 2", [("import", None), ("function", "twice")]),
        ("a.js", "javascript", 'import x from "x";\nexport function twice(x) { return x * 2; }',
         [("import", None), ("function", "twice")]),
        ("a.ts", "typescript", 'interface Shape { area(): number; }\nexport const twice = (x: number) => x * 2;',
         [("interface", "Shape"), ("function", "twice")]),
        ("a.tsx", "tsx", 'export function View() { return <div>Hello</div>; }', [("function", "View")]),
        ("A.java", "java", 'import java.util.List;\nclass Example { int value() { return 1; } }',
         [("import", None), ("class", "Example")]),
        ("a.c", "c", '#include <stdio.h>\nint twice(int x) { return x * 2; }', [("import", None), ("function", "twice")]),
        ("a.cpp", "cpp", 'struct Shape { int x; };\nint *identity(int *x) { return x; }',
         [("struct", "Shape"), ("function", "identity")]),
        ("A.cs", "csharp", 'using System;\ninterface Shape { int Area(); }', [("import", None), ("interface", "Shape")]),
        ("a.go", "go", 'package a\nimport "fmt"\ntype Shape struct { X int }\nfunc twice(x int) int { return x * 2 }',
         [("declaration", None), ("import", None), ("struct", "Shape"), ("function", "twice")]),
        ("a.rs", "rust", 'use std::fmt;\nstruct Shape { x: i32 }\nfn twice(x: i32) -> i32 { x * 2 }',
         [("import", None), ("struct", "Shape"), ("function", "twice")]),
        ("a.rb", "ruby", "class Shape\n  def area\n    42\n  end\nend", [("class", "Shape")]),
        ("a.php", "php", "<?php\nfunction twice($x) { return $x * 2; }", [("code", None), ("function", "twice")]),
    ]
    for path, language, code, expected in examples:
        chunks = chunks_for(path, code)
        assert [(c.chunk_type, c.symbol_name) for c in chunks] == expected, path
        assert {c.language for c in chunks} == {language}
        assert {c.parser for c in chunks} == {"tree-sitter"}


def test_hierarchy():
    code = "class Example:\n    def one(self): return 1\n    def two(self): return 2"
    assert chunks_for("a.py", code)[0].content == code
    methods = [c for c in chunks_for("a.py", code, 40) if c.chunk_type == "method"]
    assert [c.symbol_name for c in methods] == ["one", "two"]
    assert {c.parent_symbol for c in methods} == {"Example"}
    nested = "class Example:\n    def outer(self):\n        def inner():\n            return 42\n        return inner()\n"
    inner = next(c for c in chunks_for("a.py", nested, 48) if c.symbol_name == "inner")
    assert inner.chunk_type == "function"
    assert inner.parent_symbol == "Example.outer"


def test_method_parents():
    for path, code, parent in [
        ("a.go", "package a\ntype Shape struct { X int }\nfunc (s *Shape) Area() int { return s.X }", "Shape"),
        ("a.rs", "impl Shape { fn area(&self) -> i32 { self.x } fn twice(&self) -> i32 { self.x * 2 } }", "Shape"),
        ("a.cs", "namespace N { class C { int First() { return 1; } int Second() { return 2; } } }", "N.C"),
    ]:
        methods = [c for c in chunks_for(path, code, 44) if c.chunk_type == "method"]
        assert methods
        assert {c.parent_symbol for c in methods} == {parent}


def test_decorators_unicode_and_crlf():
    code = "@decorate('тест')\r\ndef привет(имя):\r\n    return 'Привет, ' + имя\r\n"
    chunk, = chunks_for("пакет/пример.py", code)
    assert chunk.symbol_name == "привет"
    assert chunk.content == code.rstrip("\r\n")
    assert (chunk.start_line, chunk.end_line) == (1, 3)


def test_structural_boundaries():
    statements = [f"    print('statement {i}')\n" for i in range(20)]
    chunks = chunks_for("a.py", "def process():\n" + "".join(statements), 100)
    assert len(chunks) > 1
    assert all(c.symbol_name == "process" for c in chunks)
    assert all(any(line.strip() in c.content for c in chunks) for line in statements)
    pairs = [f'"key{i}": "value-{i}"' for i in range(8)]
    chunks = chunks_for("package.json", "{" + ", ".join(pairs) + "}", 40)
    assert all(any(pair in c.content for c in chunks) for pair in pairs)
    assert chunks_for("a.c", "struct Shape { int x; };\n")[0].content.endswith(";")


def test_fallback_and_long_lines(caplog):
    for path, limit in [("a.unknown", 1), ("a.py", 17), ("config.json", 64)]:
        chunks = chunks_for(path, 'value = "' + "привет🙂" * 100 + '"\n', limit)
        assert {(c.start_line, c.end_line) for c in chunks} == {(1, 1)}
    for path in ("a.unknown", "notes.txt"):
        chunks = chunks_for(path, "".join(f"line-{i:02}\n" for i in range(10)), 24, overlap=1)
        assert all(c.parser == "lines" for c in chunks)
        assert chunks[0].content.splitlines()[-1] == chunks[1].content.splitlines()[0]
        assert chunks[1].end_line > chunks[0].end_line

    def unavailable(language):
        raise RuntimeError("grammar unavailable")
    chunks = chunks_for("a.py", "def f(): return 42\n" * 20, 50, parser_factory=unavailable)
    assert {(c.language, c.parser) for c in chunks} == {("python", "lines")}
    assert "grammar unavailable" in caplog.text


def test_configs():
    for path, content, language in [
        ("pyproject.toml", '[project]\nname = "example"\n[tool.pytest]\ntestpaths = ["tests"]\n', "toml"),
        ("package.json", '{"scripts": {"test": "jest"}, "type": "module"}', "json"),
        ("workflow.yml", "jobs:\n  test:\n    steps:\n      - run: pytest\n", "yaml"),
        ("requirements-dev.txt", "pytest==9.1.1\nhttpx==0.28.1\n", "text"),
        (".editorconfig", "root = true\n[*]\nindent_size = 4\n", "ini"),
    ]:
        assert {(c.chunk_type, c.language) for c in chunks_for(path, content, 35)} == {("config", language)}


def test_registry_extension_and_detection():
    registry = LanguageRegistry()
    assert registry.detect("scripts/run", "#!/usr/bin/env python3\nprint(1)") == ("python", False)
    assert registry.detect(r"C:\project\Example.ts", "") == ("typescript", False)
    registry.register("custom", LanguageProfile(node_types={
        **COMMON_NODE_TYPES, "function_definition": "custom_function",
    }), extensions=(".custom",))
    chunk, = chunks_for("a.custom", "def example(): return 1", registry=registry,
                        parser_factory=lambda _: get_parser("python"))
    assert (chunk.language, chunk.chunk_type, chunk.symbol_name) == ("custom", "custom_function", "example")


def test_partial_empty_and_invalid_inputs():
    for code in ["import os\ndef broken(:\n return 1\ndef valid(): return 2", "def f():\n value = (", "@@@\n def", "# comment"]:
        chunks_for("a.py", code, 30)
    for content in ("", " \n\t", "binary\x00data"):
        assert ProjectChunker().chunk_file(ProjectFile(path="a.py", content=content)) == []
    for options in ({"max_chunk_chars": 0}, {"max_chunk_chars": -1}, {"overlap_lines": -1}):
        with pytest.raises(ValueError):
            ChunkerConfig(**options)


def test_project_identity():
    files = [ProjectFile(path=f"{name}/a.py", content="def f(): return 1") for name in ("one", "two")]
    chunker = ProjectChunker()
    chunks = chunker.chunk_project(files)
    assert chunks == chunker.chunk_project(iter(files))
    assert [c.file_path for c in chunks] == [f.path for f in files]
    assert chunker.chunk_project([]) == []
