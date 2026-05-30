import httpx
import json
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import List, Optional
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_random_exponential
from tenacity.wait import wait_base
from zoneinfo import ZoneInfo
from .base import BaseSearchProvider, SearchResult
from ..utils import search_prompt, fetch_prompt, url_describe_prompt, rank_sources_prompt, redact_sensitive_text
from ..logger import log_info
from ..config import config


def get_local_time_info() -> str:
    """获取本地时间信息，用于注入到搜索查询中"""
    try:
        # 尝试获取系统本地时区
        local_tz = datetime.now().astimezone().tzinfo
        local_now = datetime.now(local_tz)
    except Exception:
        # 降级使用 UTC
        local_now = datetime.now(timezone.utc)

    # 格式化时间信息
    weekdays_cn = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    weekday = weekdays_cn[local_now.weekday()]

    return (
        f"[Current Time Context]\n"
        f"- Date: {local_now.strftime('%Y-%m-%d')} ({weekday})\n"
        f"- Time: {local_now.strftime('%H:%M:%S')}\n"
        f"- Timezone: {local_now.tzname() or 'Local'}\n"
    )


def _needs_time_context(query: str) -> bool:
    """检查查询是否需要时间上下文"""
    # 中文时间相关关键词
    cn_keywords = [
        "当前", "现在", "今天", "明天", "昨天",
        "本周", "上周", "下周", "这周",
        "本月", "上月", "下月", "这个月",
        "今年", "去年", "明年",
        "最新", "最近", "近期", "刚刚", "刚才",
        "实时", "即时", "目前",
    ]
    # 英文时间相关关键词
    en_keywords = [
        "current", "now", "today", "tomorrow", "yesterday",
        "this week", "last week", "next week",
        "this month", "last month", "next month",
        "this year", "last year", "next year",
        "latest", "recent", "recently", "just now",
        "real-time", "realtime", "up-to-date",
    ]

    query_lower = query.lower()

    for keyword in cn_keywords:
        if keyword in query:
            return True

    for keyword in en_keywords:
        if keyword in query_lower:
            return True

    return False

RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


def _split_csv_values(value: str) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def _is_x_platform(platform: str) -> bool:
    normalized = (platform or "").strip().lower()
    return normalized in {"x", "twitter", "x/twitter", "twitter/x", "推特", "x平台"}


def _is_multi_agent_model(model: str) -> bool:
    return "grok-4.20-multi-agent" in (model or "").lower()


def _normalize_multi_agent_model(model: str) -> tuple[str, str]:
    normalized = (model or "").strip()
    lower = normalized.lower()
    for suffix, effort in (
        ("-xhigh", "xhigh"),
        ("-high", "high"),
        ("-medium", "medium"),
        ("-low", "low"),
    ):
        if lower.endswith(suffix):
            return normalized[: -len(suffix)], effort
    return normalized, ""


def _valid_reasoning_effort(effort: str) -> str:
    normalized = (effort or "").strip().lower()
    return normalized if normalized in {"low", "medium", "high", "xhigh"} else ""


def _is_retryable_exception(exc) -> bool:
    """检查异常是否可重试"""
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.ConnectError, httpx.RemoteProtocolError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS_CODES
    return False


class _WaitWithRetryAfter(wait_base):
    """等待策略：优先使用 Retry-After 头，否则使用指数退避"""

    def __init__(self, multiplier: float, max_wait: int):
        self._base_wait = wait_random_exponential(multiplier=multiplier, max=max_wait)
        self._protocol_error_base = 3.0

    def __call__(self, retry_state):
        if retry_state.outcome and retry_state.outcome.failed:
            exc = retry_state.outcome.exception()
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
                retry_after = self._parse_retry_after(exc.response)
                if retry_after is not None:
                    return retry_after
            if isinstance(exc, httpx.RemoteProtocolError):
                return self._base_wait(retry_state) + self._protocol_error_base
        return self._base_wait(retry_state)

    def _parse_retry_after(self, response: httpx.Response) -> Optional[float]:
        """解析 Retry-After 头（支持秒数或 HTTP 日期格式）"""
        header = response.headers.get("Retry-After")
        if not header:
            return None
        header = header.strip()

        if header.isdigit():
            return float(header)

        try:
            retry_dt = parsedate_to_datetime(header)
            if retry_dt.tzinfo is None:
                retry_dt = retry_dt.replace(tzinfo=timezone.utc)
            delay = (retry_dt - datetime.now(timezone.utc)).total_seconds()
            return max(0.0, delay)
        except (TypeError, ValueError):
            return None


class GrokSearchProvider(BaseSearchProvider):
    def __init__(
        self,
        api_url: str,
        api_key: str,
        model: str = "grok-4-fast",
        reasoning_effort: str = "",
    ):
        super().__init__(api_url, api_key)
        self.model = model
        self.responses_model, model_effort = _normalize_multi_agent_model(model)
        self.reasoning_effort = _valid_reasoning_effort(reasoning_effort) or model_effort

    def get_provider_name(self) -> str:
        return "Grok"

    def _build_search_payload(
        self,
        query: str,
        platform: str = "",
        from_date: str = "",
        to_date: str = "",
        allowed_domains: str = "",
        max_search_results: int = 0,
    ) -> dict:
        prompt_lines: list[str] = []
        payload: dict = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": search_prompt,
                },
            ],
            "stream": True,
        }

        if _is_x_platform(platform):
            payload["tools"] = [{"type": "x_search"}]
            prompt_lines.append("Use X/Twitter search for this query and include x.com links when relevant.")
        elif platform:
            prompt_lines.append(f"Focus the search on this platform or source type: {platform}.")

        search_parameters: dict = {}
        if from_date:
            search_parameters["from_date"] = from_date
            prompt_lines.append(f"Only use results dated on or after {from_date}.")
        if to_date:
            search_parameters["to_date"] = to_date
            prompt_lines.append(f"Only use results dated on or before {to_date}.")
        if search_parameters:
            search_parameters["mode"] = "on"
            payload["search_parameters"] = search_parameters

        domains = _split_csv_values(allowed_domains)
        if domains:
            prompt_lines.append(
                "Only search and cite these domains: " + ", ".join(domains) + "."
            )

        if max_search_results > 0:
            prompt_lines.append(
                f"Use no more than {max_search_results} high-quality search results or citations in the final answer."
            )

        time_context = get_local_time_info() + "\n"
        controls = ("\n\n[Search Controls]\n" + "\n".join(f"- {line}" for line in prompt_lines)) if prompt_lines else ""
        payload["messages"].append({"role": "user", "content": time_context + query + controls})
        return payload

    def _build_responses_search_payload(
        self,
        query: str,
        platform: str = "",
        from_date: str = "",
        to_date: str = "",
        allowed_domains: str = "",
        max_search_results: int = 0,
        reasoning_effort: str = "",
    ) -> dict:
        prompt_lines: list[str] = []
        time_context = get_local_time_info()
        tools: list[dict] = []

        web_tool: dict = {"type": "web_search"}
        domains = _split_csv_values(allowed_domains)
        if domains:
            web_tool["filters"] = {"allowed_domains": domains[:5]}
            prompt_lines.append(
                "Only search and cite these domains: " + ", ".join(domains[:5]) + "."
            )

        if _is_x_platform(platform):
            tools.extend([web_tool, {"type": "x_search"}])
            prompt_lines.append("Use both web_search and X/Twitter x_search when useful.")
        else:
            tools.append(web_tool)
            if platform:
                prompt_lines.append(f"Focus the search on this platform or source type: {platform}.")

        if from_date:
            prompt_lines.append(f"Only use results dated on or after {from_date}.")
        if to_date:
            prompt_lines.append(f"Only use results dated on or before {to_date}.")
        if max_search_results > 0:
            prompt_lines.append(
                f"Use no more than {max_search_results} high-quality search results or citations in the final answer."
            )

        controls = (
            "\n\n[Search Controls]\n" + "\n".join(f"- {line}" for line in prompt_lines)
            if prompt_lines
            else ""
        )
        payload: dict = {
            "model": self.responses_model,
            "input": [
                {
                    "role": "user",
                    "content": f"{search_prompt}\n\n{time_context}\n{query}{controls}",
                }
            ],
            "tools": tools,
            "stream": True,
        }
        effective_effort = _valid_reasoning_effort(reasoning_effort) or self.reasoning_effort
        if effective_effort:
            payload["reasoning"] = {"effort": effective_effort}
        return payload

    async def search(
        self,
        query: str,
        platform: str = "",
        min_results: int = 3,
        max_results: int = 10,
        ctx=None,
        from_date: str = "",
        to_date: str = "",
        allowed_domains: str = "",
        max_search_results: int = 0,
        reasoning_effort: str = "",
    ) -> List[SearchResult]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = self._build_search_payload(
            query=query,
            platform=platform,
            from_date=from_date,
            to_date=to_date,
            allowed_domains=allowed_domains,
            max_search_results=max_search_results,
        )

        if _is_multi_agent_model(self.model):
            payload = self._build_responses_search_payload(
                query=query,
                platform=platform,
                from_date=from_date,
                to_date=to_date,
                allowed_domains=allowed_domains,
                max_search_results=max_search_results,
                reasoning_effort=reasoning_effort,
            )

        await log_info(ctx, f"search_payload: {redact_sensitive_text(json.dumps(payload, ensure_ascii=False), self.api_key)}", config.debug_enabled)

        if _is_multi_agent_model(self.model):
            try:
                return await self._execute_responses_stream_with_retry(headers, payload, ctx)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in {404, 405}:
                    raise
                await log_info(ctx, "responses_api_unavailable: falling back to chat/completions", config.debug_enabled)
                payload = self._build_search_payload(
                    query=query,
                    platform=platform,
                    from_date=from_date,
                    to_date=to_date,
                    allowed_domains=allowed_domains,
                    max_search_results=max_search_results,
                )

        return await self._execute_stream_with_retry(headers, payload, ctx)

    async def fetch(self, url: str, ctx=None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": fetch_prompt,
                },
                {"role": "user", "content": url + "\n获取该网页内容并返回其结构化Markdown格式" },
            ],
            "stream": True,
        }
        return await self._execute_stream_with_retry(headers, payload, ctx)

    async def _parse_streaming_response(self, response, ctx=None) -> str:
        content = ""
        full_body_buffer = []
        parse_errors = 0
        
        async for line in response.aiter_lines():
            line = line.strip()
            if not line:
                continue
            
            full_body_buffer.append(line)

            # 兼容 "data: {...}" 和 "data:{...}" 两种 SSE 格式
            if line.startswith("data:"):
                if line in ("data: [DONE]", "data:[DONE]"):
                    continue
                try:
                    # 去掉 "data:" 前缀，并去除可能的空格
                    json_str = line[5:].lstrip()
                    data = json.loads(json_str)
                    choices = data.get("choices", [])
                    if choices and len(choices) > 0:
                        delta = choices[0].get("delta", {})
                        if "content" in delta:
                            content += delta["content"]
                except (json.JSONDecodeError, IndexError):
                    parse_errors += 1
                    continue
                
        if not content and full_body_buffer:
            try:
                full_text = "".join(full_body_buffer)
                data = json.loads(full_text)
                if "choices" in data and len(data["choices"]) > 0:
                    message = data["choices"][0].get("message", {})
                    content = message.get("content", "")
            except json.JSONDecodeError:
                parse_errors += 1

        if not content:
            if full_body_buffer:
                snippet = redact_sensitive_text("\n".join(full_body_buffer[:5]), self.api_key).strip()
                if len(snippet) > 500:
                    snippet = snippet[:500] + "..."
                raise ValueError(
                    "Grok stream parse error: no content found in streaming response"
                    f" (parse_errors={parse_errors}, snippet={snippet})"
                )
            raise ValueError("Grok stream parse error: empty streaming response")
        
        await log_info(ctx, f"content: {content}", config.debug_enabled)

        return content

    def _extract_responses_text_and_citations(self, data: dict) -> tuple[str, list[str]]:
        if not isinstance(data, dict):
            return "", []

        if isinstance(data.get("response"), dict):
            response_text, response_citations = self._extract_responses_text_and_citations(data["response"])
            if response_text or response_citations:
                return response_text, response_citations

        text_parts: list[str] = []
        citations: list[str] = []

        output_text = data.get("output_text")
        if isinstance(output_text, str):
            text_parts.append(output_text)

        output = data.get("output")
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue
                for content_item in item.get("content", []) or []:
                    if not isinstance(content_item, dict):
                        continue
                    if content_item.get("type") == "output_text" and isinstance(content_item.get("text"), str):
                        text_parts.append(content_item["text"])
                    citations.extend(self._extract_citation_urls(content_item.get("annotations")))

        citations.extend(self._extract_citation_urls(data.get("citations")))
        citations.extend(self._extract_citation_urls(data.get("inline_citations")))
        return "".join(text_parts), self._dedupe_urls(citations)

    def _extract_citation_urls(self, value) -> list[str]:
        urls: list[str] = []
        if isinstance(value, str):
            if value.startswith(("http://", "https://")):
                urls.append(value)
            return urls
        if isinstance(value, list):
            for item in value:
                urls.extend(self._extract_citation_urls(item))
            return urls
        if isinstance(value, dict):
            url = value.get("url")
            if isinstance(url, str) and url.startswith(("http://", "https://")):
                urls.append(url)
            for key in ("web_citation", "x_citation", "citations", "annotations", "inline_citations"):
                if key in value:
                    urls.extend(self._extract_citation_urls(value[key]))
        return urls

    def _dedupe_urls(self, urls: list[str]) -> list[str]:
        seen: set[str] = set()
        unique: list[str] = []
        for url in urls:
            cleaned = (url or "").strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            unique.append(cleaned)
        return unique

    def _append_sources_footer(self, content: str, citations: list[str]) -> str:
        unique = self._dedupe_urls(citations)
        if not unique:
            return content
        footer = "\n".join(f"[[{idx}]]({url})" for idx, url in enumerate(unique, start=1))
        return f"{content.rstrip()}\n\nSources:\n{footer}"

    async def _parse_responses_streaming_response(self, response, ctx=None) -> str:
        content = ""
        citations: list[str] = []
        full_body_buffer = []
        parse_errors = 0

        async for line in response.aiter_lines():
            line = line.strip()
            if not line or line.startswith("event:"):
                continue

            full_body_buffer.append(line)
            json_str = line[5:].lstrip() if line.startswith("data:") else line
            if json_str in ("[DONE]", ""):
                continue

            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                parse_errors += 1
                continue

            event_type = data.get("type")
            if event_type == "response.output_text.delta" and isinstance(data.get("delta"), str):
                content += data["delta"]
                continue

            response_text, response_citations = self._extract_responses_text_and_citations(data)
            if response_text and not content:
                content = response_text
            citations.extend(response_citations)

        if not content and full_body_buffer:
            try:
                full_text = "".join(
                    line[5:].lstrip() if line.startswith("data:") else line
                    for line in full_body_buffer
                    if line not in ("data: [DONE]", "data:[DONE]")
                )
                data = json.loads(full_text)
                content, citations = self._extract_responses_text_and_citations(data)
            except json.JSONDecodeError:
                parse_errors += 1

        if not content:
            if full_body_buffer:
                snippet = redact_sensitive_text("\n".join(full_body_buffer[:5]), self.api_key).strip()
                if len(snippet) > 500:
                    snippet = snippet[:500] + "..."
                raise ValueError(
                    "Grok responses stream parse error: no content found in streaming response"
                    f" (parse_errors={parse_errors}, snippet={snippet})"
                )
            raise ValueError("Grok responses stream parse error: empty streaming response")

        content = self._append_sources_footer(content, citations)
        await log_info(ctx, f"content: {content}", config.debug_enabled)
        return content

    async def _execute_stream_with_retry(self, headers: dict, payload: dict, ctx=None) -> str:
        """执行带重试机制的流式 HTTP 请求"""
        timeout = httpx.Timeout(connect=6.0, read=120.0, write=10.0, pool=None)

        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(config.retry_max_attempts + 1),
                wait=_WaitWithRetryAfter(config.retry_multiplier, config.retry_max_wait),
                retry=retry_if_exception(_is_retryable_exception),
                reraise=True,
            ):
                with attempt:
                    async with client.stream(
                        "POST",
                        f"{self.api_url}/chat/completions",
                        headers=headers,
                        json=payload,
                    ) as response:
                        if response.is_error:
                            await response.aread()
                        response.raise_for_status()
                        return await self._parse_streaming_response(response, ctx)

    async def _execute_responses_stream_with_retry(self, headers: dict, payload: dict, ctx=None) -> str:
        """执行 Responses API 流式 HTTP 请求。"""
        timeout = httpx.Timeout(connect=6.0, read=240.0, write=10.0, pool=None)

        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(config.retry_max_attempts + 1),
                wait=_WaitWithRetryAfter(config.retry_multiplier, config.retry_max_wait),
                retry=retry_if_exception(_is_retryable_exception),
                reraise=True,
            ):
                with attempt:
                    async with client.stream(
                        "POST",
                        f"{self.api_url}/responses",
                        headers=headers,
                        json=payload,
                    ) as response:
                        if response.is_error:
                            await response.aread()
                        response.raise_for_status()
                        return await self._parse_responses_streaming_response(response, ctx)

    async def describe_url(self, url: str, ctx=None) -> dict:
        """让 Grok 阅读单个 URL 并返回 title + extracts"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": url_describe_prompt},
                {"role": "user", "content": url},
            ],
            "stream": True,
        }
        result = await self._execute_stream_with_retry(headers, payload, ctx)
        title, extracts = url, ""
        for line in result.strip().splitlines():
            if line.startswith("Title:"):
                title = line[6:].strip() or url
            elif line.startswith("Extracts:"):
                extracts = line[9:].strip()
        return {"title": title, "extracts": extracts, "url": url}

    async def rank_sources(self, query: str, sources_text: str, total: int, ctx=None) -> list[int]:
        """让 Grok 按查询相关度对信源排序，返回排序后的序号列表"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": rank_sources_prompt},
                {"role": "user", "content": f"Query: {query}\n\n{sources_text}"},
            ],
            "stream": True,
        }
        result = await self._execute_stream_with_retry(headers, payload, ctx)
        order: list[int] = []
        seen: set[int] = set()
        for token in result.strip().split():
            try:
                n = int(token)
                if 1 <= n <= total and n not in seen:
                    seen.add(n)
                    order.append(n)
            except ValueError:
                continue
        # 补齐遗漏的序号
        for i in range(1, total + 1):
            if i not in seen:
                order.append(i)
        return order
