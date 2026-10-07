from app.llm import remove_comments


def test_comments_are_removed_without_changing_strings():
    examples = [
        ("python", '# описание\ndef test_value():\n    assert "# value" == "# value"  # inline\n'),
        ("javascript", '// описание\nconst url = "https://example.com"; /* block */\n'),
        ("java", 'class Test { /* описание */ String text = "// value"; }'),
    ]
    for language, code in examples:
        result = remove_comments(code, language)
        assert "описание" not in result
        assert "inline" not in result and "block" not in result
    assert '"# value"' in remove_comments(examples[0][1], "python")
    assert '"https://example.com"' in remove_comments(examples[1][1], "javascript")
    assert '"// value"' in remove_comments(examples[2][1], "java")
    assert "return value" in remove_comments("function f() { return/*comment*/value; }", "javascript")
    assert "описание" not in remove_comments(examples[0][1], "python", "script")
