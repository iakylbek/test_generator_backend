# RAGTest Backend

Backend для расширения VS Code, которое генерирует unit-тесты с помощью LLM и RAG.
Файлы проекта разбиваются на чанки через Tree-sitter, преобразуются в векторы
моделью Qwen3-Embedding-0.6B и сохраняются в локальной базе Qdrant.
При генерации связанный код добавляется в запрос к LLM через Groq API.

## Запуск

Из папки `test_generator_backend` создайте виртуальное окружение:

```bash
python -m venv .venv
```

Активируйте его:

- Windows: `.venv\Scripts\activate`
- Linux/macOS: `source .venv/bin/activate`

Установите зависимости:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Создайте в этой папке файл `.env` с ключом [Groq](https://console.groq.com/):

```env
GROQ_API_KEY=your_groq_api_key
```

Запустите сервер:

```bash
python -m uvicorn main:app --reload
```

Адрес backend: `http://127.0.0.1:8000`.
Документация и проверка запросов: `http://127.0.0.1:8000/docs`.
При первой индексации скачиваются embedding-модель и нужные грамматики Tree-sitter;
потребуются интернет и время на загрузку. Qdrant запускается внутри backend.

## Эндпоинты

| Метод и путь | Назначение |
|---|---|
| `GET /health` | Проверка работы сервера. Возвращает `{"status":"ok"}`. |
| `POST /index-project` | Принимает `project_name`, `project_id` и `project_files` с полями `path` и `content`. Создаёт чанки, вычисляет embedding и сохраняет индекс. Возвращает число файлов и чанков, а также сами чанки. |
| `POST /generate-tests` | Принимает `code`, `language`, `source_file`, `project_id` и `instruction`. Возвращает код тестов. Поле `use_rag` включает поиск контекста; по умолчанию `true`. |

Для RAG используйте одинаковый `project_id` при индексации и генерации.
При повторной индексации передавайте полный список файлов: отсутствующие чанки удаляются.
Для генерации по текстовому запросу передайте пустой `code`, `language: "auto"`,
`project_id` и `instruction`. Для генерации без RAG передайте исходный код и `use_rag: false`.
