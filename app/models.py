from pydantic import BaseModel


class TestGenerationRequest(BaseModel):
    code: str
    language: str
    framework: str | None = None
    instruction: str | None = None


class TestGenerationResponse(BaseModel):
    tests: str
