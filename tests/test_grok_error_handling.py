import pytest
import httpx

from grok_search.providers.grok import GrokSearchProvider
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

        async def search(self, query, platform):
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
