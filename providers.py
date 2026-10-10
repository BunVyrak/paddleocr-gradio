"""Vision-model backends the annotator can talk to.

Every provider exposes the same two calls:
    list_models() -> list[str]
    chat(img, prompt, model, max_tokens) -> str

Add a new backend by subclassing `Provider` and registering it in `PROVIDERS`.
"""

import base64
import io

import httpx
from openai import OpenAI
from PIL import Image


def _png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class Provider:
    default_url = ""
    needs_key = False

    def __init__(self, base_url: str = "", api_key: str = "", timeout: float = 600):
        self.base_url = (base_url or self.default_url).rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def list_models(self) -> list[str]:
        raise NotImplementedError

    def chat(self, img: Image.Image, prompt: str, model: str, max_tokens: int) -> str:
        raise NotImplementedError


class OpenAICompatProvider(Provider):
    """Any server speaking the OpenAI chat-completions API with image_url parts
    (LM Studio, vLLM, llama.cpp server, OpenRouter, ...)."""

    default_url = "http://localhost:1234/v1"

    def _client(self) -> OpenAI:
        return OpenAI(base_url=self.base_url, api_key=self.api_key or "not-needed",
                      timeout=self.timeout)

    def list_models(self) -> list[str]:
        return sorted(m.id for m in self._client().models.list().data)

    def chat(self, img, prompt, model, max_tokens):
        resp = self._client().chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=int(max_tokens),
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": "data:image/png;base64," + _png_b64(img)}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return (resp.choices[0].message.content or "").strip()


class LMStudioProvider(OpenAICompatProvider):
    default_url = "http://localhost:1234/v1"


class OllamaProvider(Provider):
    """Ollama's native /api/chat. Used instead of its OpenAI shim so we can raise
    `num_ctx` — the default context is too small for full-page images."""

    default_url = "http://localhost:11434"

    def __init__(self, base_url="", api_key="", timeout=600, num_ctx=16384):
        super().__init__(base_url, api_key, timeout)
        # Accept the OpenAI-style URL too.
        self.base_url = self.base_url.removesuffix("/v1")
        self.num_ctx = num_ctx

    def _headers(self):
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def list_models(self) -> list[str]:
        r = httpx.get(f"{self.base_url}/api/tags", headers=self._headers(), timeout=10)
        r.raise_for_status()
        return sorted(m["name"] for m in r.json().get("models", []))

    def chat(self, img, prompt, model, max_tokens):
        r = httpx.post(
            f"{self.base_url}/api/chat",
            headers=self._headers(),
            timeout=self.timeout,
            json={
                "model": model,
                "stream": False,
                "messages": [{"role": "user", "content": prompt, "images": [_png_b64(img)]}],
                "options": {"temperature": 0, "num_predict": int(max_tokens),
                            "num_ctx": self.num_ctx},
            },
        )
        if r.status_code >= 400:
            raise RuntimeError(f"Ollama {r.status_code}: {r.text.strip()}")
        return (r.json().get("message", {}).get("content") or "").strip()


PROVIDERS: dict[str, type[Provider]] = {
    "LM Studio": LMStudioProvider,
    "Ollama": OllamaProvider,
    "OpenAI-compatible": OpenAICompatProvider,
}


def get_provider(name: str, base_url: str = "", api_key: str = "") -> Provider:
    if name not in PROVIDERS:
        raise ValueError(f"Unknown provider {name!r}; choose from {list(PROVIDERS)}")
    return PROVIDERS[name](base_url, api_key)
