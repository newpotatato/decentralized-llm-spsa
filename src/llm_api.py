import time
import requests
from typing import Dict, Any, Optional


class LLMAPIClient:
    """
    OpenAI-compatible chat completions client.

    Works with any provider that implements /v1/chat/completions:
      - OpenRouter  (one key for all models)
      - Groq        (Llama, Mistral — free tier available)
      - Together AI (Qwen, Llama, DeepSeek)
      - DeepSeek    (deepseek-coder, deepseek-chat)
      - Mistral AI  (mistral-7b, mixtral)

    Parameters
    ----------
    endpoints : {model_name: base_url}
        Base URL up to /v1, e.g. "https://openrouter.ai/api/v1".
        For OpenRouter you can point every model at the same URL.
    api_keys : {model_name: key}
        API keys per model. For OpenRouter all keys are identical.
    model_aliases : {model_name: remote_model_id}, optional
        Maps internal model names to provider model IDs, e.g.
        {"llama3-8b-instruct": "meta-llama/llama-3-8b-instruct"}.
    max_tokens : int
        Max tokens per completion (default 512).
    timeout : int
        HTTP timeout in seconds (default 60).
    """

    def __init__(
        self,
        endpoints: Dict[str, str],
        api_keys: Dict[str, str],
        model_aliases: Optional[Dict[str, str]] = None,
        max_tokens: int = 512,
        max_tokens_per_model: Optional[Dict[str, int]] = None,
        timeout: int = 60,
    ):
        self.endpoints = endpoints
        self.api_keys = api_keys
        self.model_aliases = model_aliases or {}
        self.max_tokens = max_tokens
        self.max_tokens_per_model = max_tokens_per_model or {}
        self.timeout = timeout

    def call(self, model_name: str, prompt: str, task_type: str) -> Dict[str, Any]:
        """
        Send a chat completion request. Returns:
            {"output": str, "success": bool, "latency": float}
        Raises ValueError if the model is not configured.
        Raises requests.HTTPError on non-2xx responses.
        """
        base_url = self.endpoints.get(model_name)
        api_key = self.api_keys.get(model_name)
        if not base_url or not api_key:
            raise ValueError(
                f"Model '{model_name}' is not configured. "
                "Add it to LLM_ENDPOINTS and LLM_API_KEYS in llm_config.py."
            )

        remote_model = self.model_aliases.get(model_name, model_name)
        url = base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": remote_model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Respond in plain text only. Do not call any functions or tools."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_tokens_per_model.get(model_name, self.max_tokens),
        }

        t0 = time.perf_counter()
        response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        latency = time.perf_counter() - t0

        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return {
            "output": content,
            "success": True,
            "latency": latency,
        }
