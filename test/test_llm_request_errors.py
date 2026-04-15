"""Regression tests for user-facing LLM error handling."""

from pathlib import Path
import sys

import httpx
import openai
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "climsight"))

import climsight_engine as engine


def _build_rate_limit_error(message: str) -> openai.RateLimitError:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(429, request=request)
    return openai.RateLimitError(
        message,
        response=response,
        body={"error": {"message": message}},
    )


def test_llm_request_wraps_rate_limit_errors(monkeypatch):
    def raise_rate_limit(*args, **kwargs):
        raise _build_rate_limit_error(
            "You exceeded your current quota. insufficient_quota"
        )

    monkeypatch.setattr(engine, "agent_llm_request", raise_rate_limit)

    with pytest.raises(engine.ClimSightLLMError, match="API quota was exceeded"):
        engine.llm_request(
            content_message="prompt",
            input_params={"user_message": "question"},
            config={"llmModeKey": "agent_llm"},
            api_key="test-key",
            api_key_local="",
            stream_handler=None,
            ipcc_rag_ready=False,
            ipcc_rag_db=None,
            general_rag_ready=False,
            general_rag_db=None,
            data_pocket=None,
            references={"references": {}, "used": []},
        )


def test_llm_request_reraises_non_llm_errors(monkeypatch):
    def raise_value_error(*args, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(engine, "agent_llm_request", raise_value_error)

    with pytest.raises(ValueError, match="boom"):
        engine.llm_request(
            content_message="prompt",
            input_params={"user_message": "question"},
            config={"llmModeKey": "agent_llm"},
            api_key="test-key",
            api_key_local="",
            stream_handler=None,
            ipcc_rag_ready=False,
            ipcc_rag_db=None,
            general_rag_ready=False,
            general_rag_db=None,
            data_pocket=None,
            references={"references": {}, "used": []},
        )
