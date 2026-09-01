from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from app.llm import LLMClient
from app.models import TestGenerationRequest, TestGenerationResponse

load_dotenv()

app = FastAPI(title="Unit Test Generator API", version="1.0.0")
llm = LLMClient()


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/generate-tests", response_model=TestGenerationResponse)
def generate_tests(request: TestGenerationRequest):
    try:
        tests = llm.generate_tests(
            code=request.code,
            language=request.language,
            framework=request.framework,
            instruction=request.instruction
        )
        return TestGenerationResponse(tests=tests)

    except Exception as error:
        print(error)
        raise HTTPException(
            status_code=500, detail="Ошибка при генерации тестов")
