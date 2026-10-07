import os

from groq import Groq
from tree_sitter_language_pack import get_parser

from .chunking_languages import LanguageRegistry
from .prompts import SYSTEM_PROMPT


def remove_comments(code: str, language: str, source_file: str | None = None) -> str:
    if source_file:
        detected = LanguageRegistry().detect(source_file, code)[0]
        language = detected if detected not in {"unknown", "text"} else language
    source = code.encode("utf-8")
    try:
        root = get_parser(language).parse(source).root_node
    except Exception:
        return code
    comments = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type == "comment" or node.type.endswith("_comment"):
            comments.append((node.start_byte, node.end_byte))
        else:
            stack.extend(node.named_children)
    parts = []
    cursor = 0
    for start, end in sorted(comments):
        parts.extend((source[cursor:start], b" " + b"\n" * source[start:end].count(b"\n")))
        cursor = end
    parts.append(source[cursor:])
    return b"".join(parts).decode("utf-8").strip() if comments else code


class LLMClient:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY не найден")
        self.client = Groq(api_key=api_key)
        self.model = "openai/gpt-oss-120b"

    def generate_tests(
        self,
        code: str,
        language: str,
        source_file: str | None,
        framework: str | None,
        instruction: str | None,
        context: str = "",
    ) -> str:
        framework_text = framework or "выбери наиболее популярный unit-test framework для указанного языка"
        instruction_text = instruction or "Сгенерируй качественные unit-тесты."
        user_prompt = f""" 
            Язык программирования: {language}
            Исходный файл: {source_file or "не указан"}
            Тестовый framework: {framework_text}
            Дополнительная инструкция: {instruction_text}
            Исходный код: {code}
            Связанный код проекта (справочный контекст, не цель тестирования):
            {context}
            """
        print("\n=== SYSTEM PROMPT ===\n", SYSTEM_PROMPT, sep="", flush=True)
        print("\n=== USER PROMPT ===\n", user_prompt, sep="", flush=True)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_completion_tokens=4000
        )
        return remove_comments(response.choices[0].message.content or "", language, source_file)
