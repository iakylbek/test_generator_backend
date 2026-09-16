from pydantic import BaseModel


class TestGenerationRequest(BaseModel):
    code: str
    language: str
    source_file: str | None = None
    framework: str | None = None
    instruction: str | None = None


class TestGenerationResponse(BaseModel):
    tests: str
