from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"
DEFAULT_TIMEOUT_SECONDS = 45
DEFAULT_CACHE_TTL_HOURS = 12


def load_dotenv_if_present(path: Path | None = None) -> None:
    """Load simple KEY=VALUE entries without adding a new dependency."""

    dotenv_path = path or (PROJECT_ROOT / ".env")
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(
        encoding="utf-8-sig"
    ).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def load_llm_config() -> dict[str, Any]:
    path = PROJECT_ROOT / "configs" / "llm_intelligence.json"
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


class DeepSeekClient:
    """Small OpenAI-compatible DeepSeek client with local response caching."""

    def __init__(self) -> None:
        load_dotenv_if_present()
        config = load_llm_config()

        self.api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        self.base_url = os.getenv(
            "DEEPSEEK_BASE_URL",
            str(config.get("base_url", DEFAULT_BASE_URL)),
        ).rstrip("/")
        self.model = os.getenv(
            "DEEPSEEK_MODEL",
            str(config.get("default_model", DEFAULT_MODEL)),
        ).strip()
        self.enabled = bool(config.get("enabled", True))
        self.timeout_seconds = int(
            config.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
        )
        self.cache_ttl_hours = float(
            config.get("cache_ttl_hours", DEFAULT_CACHE_TTL_HOURS)
        )
        self.max_retries = int(config.get("max_retries", 2))
        self.cache_dir = (
            PROJECT_ROOT / "data" / "cache" / "deepseek"
        )

    @property
    def available(self) -> bool:
        return self.enabled and bool(self.api_key) and bool(self.model)

    def unavailable_reason(self) -> str:
        if not self.enabled:
            return "disabled_by_config"
        if not self.api_key:
            return "missing_api_key"
        if not self.model:
            return "missing_model"
        return "available"

    def _cache_key(self, payload: dict[str, Any]) -> str:
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _read_cache(self, key: str) -> dict[str, Any] | None:
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        age_seconds = time.time() - path.stat().st_mtime
        if age_seconds > self.cache_ttl_hours * 3600:
            return None
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return cached if isinstance(cached, dict) else None

    def _write_cache(self, key: str, value: dict[str, Any]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / f"{key}.json"
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def json_completion(
        self,
        *,
        system_prompt: str,
        user_payload: dict[str, Any],
        temperature: float = 0.1,
        max_tokens: int = 1800,
    ) -> dict[str, Any]:
        if not self.available:
            raise RuntimeError(
                "DeepSeek不可用：" + self.unavailable_reason()
            )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        "请只返回合法JSON。输入JSON如下：\n"
                        + json.dumps(user_payload, ensure_ascii=False)
                    ),
                },
            ],
            "response_format": {"type": "json_object"},
            # DeepSeek当前默认开启思考模式。结构化JSON任务明确使用
            # 非思考模式，可避免reasoning输出干扰最终JSON正文。
            "thinking": {"type": "disabled"},
            "reasoning_effort": "none",
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        cache_key = self._cache_key(payload)
        cached = self._read_cache(cache_key)
        if cached is not None:
            cached["_deepseek_cache_hit"] = True
            return cached

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                request_payload = dict(payload)
                request_payload["max_tokens"] = min(
                    max_tokens * (attempt + 1),
                    12000,
                )

                if attempt > 0:
                    request_payload["messages"] = [
                        payload["messages"][0],
                        {
                            "role": "user",
                            "content": (
                                payload["messages"][1]["content"]
                                + "\n\n上一次响应不是完整合法JSON。"
                                "这次不要输出解释、Markdown或代码围栏，"
                                "必须从{开始并以}结束。"
                            ),
                        },
                    ]

                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=request_payload,
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                body = response.json()
                choice = body["choices"][0]
                finish_reason = choice.get("finish_reason")
                content = choice["message"].get("content")

                if finish_reason != "stop":
                    raise ValueError(
                        "DeepSeek未正常结束输出："
                        f"finish_reason={finish_reason}"
                    )

                if not isinstance(content, str) or not content.strip():
                    raise ValueError(
                        "DeepSeek JSON正文为空；接口可能将无法识别的"
                        "非JSON输出替换为空格。"
                    )

                try:
                    result = json.loads(content.strip())
                except json.JSONDecodeError as error:
                    raise ValueError(
                        "DeepSeek返回了不完整JSON："
                        f"finish_reason={finish_reason}, "
                        f"content_length={len(content)}, "
                        f"json_error={error}"
                    ) from error
                if not isinstance(result, dict):
                    raise ValueError("DeepSeek返回的JSON不是对象。")
                result["_deepseek_cache_hit"] = False
                result["_deepseek_model"] = body.get(
                    "model", self.model
                )
                result["_deepseek_received_at_utc"] = datetime.now(
                    timezone.utc
                ).isoformat()
                self._write_cache(cache_key, result)
                return result
            except (
                requests.RequestException,
                KeyError,
                IndexError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as error:
                last_error = error
                if attempt < self.max_retries:
                    time.sleep(1.5 * (attempt + 1))

        raise RuntimeError(
            f"DeepSeek请求失败：{type(last_error).__name__}: {last_error}"
        ) from last_error

