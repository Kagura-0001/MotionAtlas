from __future__ import annotations

import base64
import os
import random
import time
from io import BytesIO
from typing import Any, Optional

import requests
from PIL import Image


class OpenAICompatibleClient:
    """Small OpenAI-compatible chat client for OpenAI, proxies, and local vLLM."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "EMPTY",
        timeout: float = 300.0,
        max_retries: int = 3,
        extra_headers: Optional[dict[str, str]] = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = float(timeout)
        self.max_retries = max(1, int(max_retries))
        self.extra_headers = dict(extra_headers or {})
        self.chat_url = f"{self.base_url}/chat/completions"

    @staticmethod
    def image_to_base64(image: Image.Image, image_format: str = "JPEG") -> str:
        if image.mode != "RGB":
            image = image.convert("RGB")
        buffer = BytesIO()
        image.save(buffer, format=image_format)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def build_content(
        self,
        prompt: str,
        images: list[Image.Image],
        image_detail: str = "auto",
        image_format: str = "JPEG",
    ) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        for index, image in enumerate(images, start=1):
            content.append({"type": "text", "text": f"Frame {index}:"})
            encoded = self.image_to_base64(image, image_format=image_format)
            mime = "image/png" if image_format.upper() == "PNG" else "image/jpeg"
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{encoded}",
                        "detail": image_detail,
                    },
                }
            )
        content.append({"type": "text", "text": prompt})
        return content

    def _post_chat(self, payload: dict[str, Any]) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        headers.update(self.extra_headers)

        last_error = "unknown error"
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    self.chat_url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                if response.status_code == 200:
                    data = response.json()
                    content = data["choices"][0]["message"].get("content")
                    if content is None:
                        raise RuntimeError(f"malformed response: {response.text[:500]}")
                    return str(content)
                last_error = f"HTTP {response.status_code}: {response.text[:500]}"
                if response.status_code < 500 and response.status_code not in {408, 429}:
                    break
            except Exception as exc:
                last_error = str(exc)

            if attempt < self.max_retries:
                fixed = os.getenv("MOTIONATLAS_API_RETRY_FIXED_WAIT_SECONDS", "").strip()
                if fixed:
                    wait = max(0.0, float(fixed))
                else:
                    wait = min(2 ** (attempt - 1) + random.random(), 30.0)
                time.sleep(wait)

        raise RuntimeError(f"chat completion failed after {self.max_retries} attempts: {last_error}")

    def chat(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        system_prompt: Optional[str] = None,
        response_format: Optional[dict[str, Any]] = None,
    ) -> str:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "stream": False,
        }
        if response_format:
            payload["response_format"] = response_format
        return self._post_chat(payload)

    def chat_with_images(
        self,
        prompt: str,
        images: list[Image.Image],
        max_tokens: int = 64,
        temperature: float = 0.0,
        top_p: float = 1.0,
        system_prompt: Optional[str] = None,
        image_detail: str = "auto",
        image_format: str = "JPEG",
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
    ) -> str:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": self.build_content(
                    prompt,
                    images,
                    image_detail=image_detail,
                    image_format=image_format,
                ),
            }
        )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "stream": False,
        }
        if mm_processor_kwargs:
            payload["mm_processor_kwargs"] = mm_processor_kwargs

        return self._post_chat(payload)
