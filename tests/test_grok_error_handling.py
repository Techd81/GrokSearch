import pytest
import httpx

from grok_search.providers.grok import GrokSearchProvider
from grok_search.sources import split_answer_and_sources
from grok_search.server import _format_grok_error, web_search
from grok_search.utils import extract_unique_urls, redact_sensitive_text


class FakeStreamingResponse:
    def __init__(self, lines):
        self._lines = lines

    async def aiter_lines(self):
        for line in self._lines:
            yield line


def test_format_grok_http_error_includes_status_and_redacts_secret():
    request = httpx.Request("POST", "https://api.example.test/v1/chat/completions")
    response = httpx.Response(
        400,
        text='{"error":"bad model","api_key":"secret-key"}',
        request=request,
    )
    exc = httpx.HTTPStatusError("bad request", request=request, response=response)

    message = _format_grok_error(exc, api_key="secret-key")

    assert "Grok 调用失败" in message
    assert "HTTP 400" in message
    assert "bad model" in message
    assert "secret-key" not in message


def test_format_grok_http_error_handles_unread_streaming_body():
    request = httpx.Request("POST", "https://api.example.test/v1/chat/completions")
    response = httpx.Response(
        404,
        stream=httpx.ByteStream(b'{"error":"missing route","api_key":"secret-key"}'),
        request=request,
    )
    exc = httpx.HTTPStatusError("not found", request=request, response=response)

    message = _format_grok_error(exc, api_key="secret-key")

    assert "Grok 调用失败" in message
    assert "HTTP 404" in message
    assert "secret-key" not in message


def test_redact_sensitive_text_masks_authorization_and_tokens():
    text = "Authorization: Bearer abc123 token=secret-token"

    redacted = redact_sensitive_text(text, api_key="secret-token")

    assert "abc123" not in redacted
    assert "secret-token" not in redacted
    assert "Bearer ***" in redacted


def test_extract_unique_urls_stops_before_citation_markup():
    text = "The official Python website is https://www.python.org/.[[1]](https://www.python.org/)"

    urls = extract_unique_urls(text)

    assert urls == ["https://www.python.org/"]


def test_split_answer_and_sources_extracts_inline_grok_citations():
    text = "The official Python website is https://www.python.org/.[[1]](https://www.python.org/)"

    answer, sources = split_answer_and_sources(text)

    assert answer == text
    assert sources == [{"url": "https://www.python.org/"}]


def test_grok_payload_uses_x_tool_and_date_parameters():
    provider = GrokSearchProvider("https://api.example.test/v1", "secret-key", "grok-4.3-high")

    payload = provider._build_search_payload(
        query="X 上关于 AI 的最新讨论",
        platform="Twitter",
        from_date="2026-05-01",
        to_date="2026-05-18",
        allowed_domains="x.com, techcrunch.com",
        max_search_results=3,
    )

    assert payload["stream"] is True
    assert payload["tools"] == [{"type": "x_search"}]
    assert payload["search_parameters"] == {
        "from_date": "2026-05-01",
        "to_date": "2026-05-18",
        "mode": "on",
    }
    user_content = payload["messages"][-1]["content"]
    assert "Only search and cite these domains: x.com, techcrunch.com." in user_content
    assert "Use no more than 3 high-quality search results" in user_content


def test_multi_agent_payload_uses_responses_api_shape_and_reasoning_effort():
    provider = GrokSearchProvider(
        "https://api.x.ai/v1",
        "secret-key",
        "grok-4.20-multi-agent-xhigh",
    )

    payload = provider._build_responses_search_payload(
        query="Research xAI release notes",
        platform="Twitter",
        allowed_domains="x.ai, docs.x.ai, x.com, example.com, openai.com, extra.test",
        max_search_results=5,
    )

    assert payload["model"] == "grok-4.20-multi-agent"
    assert payload["stream"] is True
    assert payload["reasoning"] == {"effort": "xhigh"}
    assert payload["tools"] == [
        {
            "type": "web_search",
            "filters": {
                "allowed_domains": ["x.ai", "docs.x.ai", "x.com", "example.com", "openai.com"],
            },
        },
        {"type": "x_search"},
    ]
    assert "input" in payload
    assert "messages" not in payload
    assert "search_parameters" not in payload


def test_multi_agent_gateway_payload_preserves_gateway_model_alias():
    provider = GrokSearchProvider(
        "https://api.example.test/v1",
        "secret-key",
        "grok-4.20-multi-agent-xhigh",
    )

    payload = provider._build_responses_search_payload(query="Reply OK")

    assert payload["model"] == "grok-4.20-multi-agent-xhigh"
    assert payload["reasoning"] == {"effort": "xhigh"}


def test_multi_agent_gateway_model_not_found_can_fallback_to_chat():
    request = httpx.Request("POST", "https://api.example.test/v1/responses")
    response = httpx.Response(
        503,
        text='{"error":{"code":"model_not_found","message":"无可用渠道"}}',
        request=request,
    )
    exc = httpx.HTTPStatusError("unavailable", request=request, response=response)
    provider = GrokSearchProvider("https://api.example.test/v1", "secret-key", "grok-4.20-multi-agent-xhigh")

    assert provider._should_fallback_from_responses_api(exc) is True


@pytest.mark.asyncio
async def test_parse_responses_streaming_response_extracts_delta_and_citations():
    provider = GrokSearchProvider("https://api.example.test/v1", "secret-key", "grok-4.20-multi-agent")
    response = FakeStreamingResponse([
        'data: {"type":"response.output_text.delta","delta":"Final answer."}',
        'data: {"type":"response.completed","response":{"citations":["https://x.ai/news"],"output":[]}}',
        "data: [DONE]",
    ])

    content = await provider._parse_responses_streaming_response(response)

    assert "Final answer." in content
    assert "Sources:" in content
    assert "[[1]](https://x.ai/news)" in content


@pytest.mark.asyncio
async def test_parse_streaming_response_raises_for_unparseable_empty_content():
    provider = GrokSearchProvider("https://api.example.test/v1", "secret-key", "bad-model")
    response = FakeStreamingResponse([
        'data: {"error":"bad stream","token":"secret-key"}',
    ])

    with pytest.raises(ValueError) as exc_info:
        await provider._parse_streaming_response(response)

    message = str(exc_info.value)
    assert "Grok stream parse error" in message
    assert "secret-key" not in message


@pytest.mark.asyncio
async def test_web_search_returns_grok_error_in_content(monkeypatch):
    class FailingProvider:
        def __init__(self, api_url, api_key, model):
            pass

        async def search(self, query, platform="", **kwargs):
            raise RuntimeError("backend failed with secret-key")

    monkeypatch.setenv("GROK_API_URL", "https://api.example.test/v1")
    monkeypatch.setenv("GROK_API_KEY", "secret-key")

    import grok_search.server as server
    monkeypatch.setattr(server, "GrokSearchProvider", FailingProvider)

    result = await web_search("Reply exactly: OK", extra_sources=0)

    assert result["content"]
    assert "Grok 调用失败" in result["content"]
    assert "backend failed" in result["content"]
    assert "secret-key" not in result["content"]
    assert result["sources_count"] == 0
