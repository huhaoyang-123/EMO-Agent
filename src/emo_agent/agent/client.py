"""DeepSeek's native function calling over its Chat Completions endpoint."""
import os

from openai import OpenAI

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def create_client(config, *, api_key=None):
    key = (api_key or os.getenv("DEEPSEEK_API_KEY", "")).strip()
    if not key:
        raise ValueError("请在本机提供 DEEPSEEK_API_KEY，或使用工具直查。")
    return OpenAI(base_url=DEEPSEEK_BASE_URL, api_key=key,
                  timeout=config["api_timeout_seconds"], max_retries=0)
