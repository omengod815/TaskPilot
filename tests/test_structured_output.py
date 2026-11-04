import pytest
from pydantic import BaseModel, Field

from app.services.llm_service import AdaptiveLLMService


class ExampleSchema(BaseModel):
    title: str
    score: int = Field(ge=0, le=100)


def test_json_parser_handles_code_fence():
    parsed = AdaptiveLLMService._parse_json('```json\n{"title":"ok","score":88}\n```')
    assert ExampleSchema.model_validate(parsed).score == 88


def test_json_parser_rejects_non_object():
    with pytest.raises(ValueError):
        AdaptiveLLMService._parse_json('[1, 2, 3]')
