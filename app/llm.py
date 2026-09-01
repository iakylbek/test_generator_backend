import os

from groq import Groq

from .prompts import SYSTEM_PROMPT


class LLMClient:

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise RuntimeError("GROQ_API_KEY не найден")

        self.client = Groq(api_key=api_key)
        self.model = "openai/gpt-oss-120b"

    def generate_tests(self, code: str, language: str, framework: str | None, instruction: str | None) -> str:
        framework_text = (
            framework
            if framework
            else "подходящий unit-test framework"
        )

        instruction_text = (
            instruction
            if instruction
            else "Сгенерируй качественные unit-тесты."
        )

        system_prompt = SYSTEM_PROMPT
        user_prompt = f""" 
            Язык программирования: {language}
            Тестовый framework: {framework_text}
            Дополнительная инструкция: {instruction_text}
            Исходный код: {code}
            """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=0.2,
            max_completion_tokens=4000
        )

        return response.choices[0].message.content or ""
