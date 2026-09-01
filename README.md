# Unit Test Generator Backend

Backend для VS Code-расширения, предназначенного для генерации unit-тестов с использованием LLM и RAG.

## Возможности

* Приём исходного кода через REST API.
* Генерация unit-тестов с помощью LLM.
* Поддержка указания языка программирования.
* Поддержка тестового framework.
* Передача дополнительной инструкции для генерации тестов.

## Технологии

* Python
* FastAPI
* Groq API
* GPT-OSS 120B
* Pydantic

## Установка

Клонировать репозиторий:

```bash
git clone <repository-url>
cd unit-test-generator-backend
```

Создать виртуальное окружение:

```bash
python -m venv .venv
```

Активировать его.

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

Установить зависимости:

```bash
pip install -r requirements.txt
```

## Настройка

Создать файл `.env` в корне проекта:

```env
GROQ_API_KEY=your_groq_api_key
```

API-ключ необходимо получить в Groq Console.

## Запуск

Запустить сервер:

```bash
uvicorn app.main:app --reload
```

После запуска API будет доступен по адресу:

```text
http://127.0.0.1:8000
```

Документация API:

```text
http://127.0.0.1:8000/docs
```