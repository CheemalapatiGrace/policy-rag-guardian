import json

import httpx
import pytest

from rag_guardian.guardrails import INSUFFICIENT
from rag_guardian.llm import AnthropicLLM, MockLLM, OllamaLLM, ResilientLLM, estimate_cost
from rag_guardian.models import Chunk
from rag_guardian.resilience import CircuitBreaker, Resilient

CH = [Chunk("d#0", "d", "Leave", "Annual", "Employees get 20 days of leave. Sick days are separate.", "x", 0, ("employee",))]


def test_mock_llm_is_extractive_and_cites():
    r = MockLLM().generate("How many days of leave?", CH)
    assert r.text == "Employees get 20 days of leave [1]." and r.input_tokens > 0 and r.output_tokens > 0
    assert MockLLM().generate("Tell me about dragons", CH).text == INSUFFICIENT


def test_mock_llm_is_deterministic():
    assert MockLLM().generate("leave days", CH).text == MockLLM().generate("leave days", CH).text


def test_anthropic_request_shape_and_parsing():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"], seen["headers"], seen["body"] = str(req.url), req.headers, json.loads(req.content)
        return httpx.Response(200, json={"content": [{"type": "text", "text": "20 days [1]."}],
                                         "usage": {"input_tokens": 120, "output_tokens": 9}})

    llm = AnthropicLLM("test-key-not-real", client=httpx.Client(transport=httpx.MockTransport(handler)))
    r = llm.generate("How many days?", CH)
    assert seen["url"] == "https://api.anthropic.com/v1/messages"
    assert seen["headers"]["x-api-key"] == "test-key-not-real" and seen["headers"]["anthropic-version"]
    assert seen["body"]["model"] == "claude-sonnet-5-5" and "<context>" in seen["body"]["messages"][0]["content"]
    assert (r.text, r.input_tokens, r.output_tokens) == ("20 days [1].", 120, 9)
    assert estimate_cost(r, 3.0, 15.0) == pytest.approx((120 * 3 + 9 * 15) / 1e6)


def test_anthropic_requires_key():
    with pytest.raises(ValueError):
        AnthropicLLM("")


def test_ollama_response_parsing():
    def handler(req):
        assert req.url.path == "/api/chat"
        return httpx.Response(200, json={"message": {"content": "ok [1]."}, "prompt_eval_count": 5, "eval_count": 2})

    r = OllamaLLM(client=httpx.Client(transport=httpx.MockTransport(handler))).generate("q", CH)
    assert r.text == "ok [1]." and r.output_tokens == 2


def test_resilient_llm_falls_back_to_extractive_when_provider_is_down():
    def handler(req):
        return httpx.Response(503)

    primary = AnthropicLLM("k", client=httpx.Client(transport=httpx.MockTransport(handler)))
    llm = ResilientLLM(primary, Resilient("llm", attempts=2, sleep=lambda s: None, breaker=CircuitBreaker(2)))
    r = llm.generate("How many days of leave?", CH)
    assert r.degraded and "20 days" in r.text
