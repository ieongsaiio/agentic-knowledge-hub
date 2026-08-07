"""OpenAI-compatible LLM implementation.

This module provides the OpenAI LLM implementation that works with
the Chat Completions or Responses API. Chat Completions can also be used with
other OpenAI-compatible endpoints by configuring the base_url.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from src.libs.llm.base_llm import BaseLLM, ChatResponse, Message


class OpenAILLMError(RuntimeError):
    """Raised when OpenAI API call fails."""


class OpenAILLM(BaseLLM):
    """OpenAI LLM provider implementation.
    
    This class implements the BaseLLM interface for OpenAI's Chat Completions
    and Responses APIs.
    
    Attributes:
        api_key: The API key for authentication.
        base_url: The base URL for the API (default: OpenAI's endpoint).
        model: The model identifier to use.
        default_temperature: Default temperature for generation.
        default_max_tokens: Default max tokens for generation.
    
    Example:
        >>> from src.core.settings import load_settings
        >>> settings = load_settings('config/settings.yaml')
        >>> llm = OpenAILLM(settings)
        >>> response = llm.chat([Message(role='user', content='Hello')])
    """
    
    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    
    def __init__(
        self,
        settings: Any,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the OpenAI LLM provider.
        
        Args:
            settings: Application settings containing LLM configuration.
            api_key: Optional API key override (falls back to settings.llm.api_key or env var).
            base_url: Optional base URL override.
            **kwargs: Additional configuration overrides.
        
        Raises:
            ValueError: If API key is not provided and not found in environment.
        
        Note:
            When azure_endpoint is present in settings, the provider automatically
            constructs the Azure-compatible OpenAI URL and uses api-key auth header.
        """
        self.model = settings.llm.model
        self.default_temperature = settings.llm.temperature
        self.default_max_tokens = settings.llm.max_tokens
        self.api_mode = str(
            getattr(settings.llm, "api_mode", "chat_completions")
        ).strip().lower()
        if self.api_mode not in {"chat_completions", "responses"}:
            raise ValueError(
                "OpenAI api_mode must be one of: chat_completions, responses"
            )
        self.extra_chat_configs = dict(
            getattr(settings.llm, "extra_chat_configs", {}) or {}
        )
        
        # API key: explicit > settings > env var
        self.api_key = (
            api_key
            or getattr(settings.llm, 'api_key', None)
            or os.environ.get("OPENAI_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. Set in settings.yaml (llm.api_key), "
                "OPENAI_API_KEY environment variable, or pass api_key parameter."
            )
        
        # Azure-compatible mode detection
        azure_endpoint = getattr(settings.llm, 'azure_endpoint', None)
        self.api_version = getattr(settings.llm, 'api_version', None)
        
        if base_url:
            self.base_url = base_url
            self._use_azure_auth = False
        elif azure_endpoint:
            # Azure-compatible mode: construct deployment-based URL
            deployment = getattr(settings.llm, 'deployment_name', None) or self.model
            self.base_url = f"{azure_endpoint.rstrip('/')}/openai/deployments/{deployment}"
            self._use_azure_auth = True
            if not self.api_version:
                self.api_version = "2024-02-15-preview"
        else:
            settings_base_url = getattr(settings.llm, 'base_url', None)
            self.base_url = settings_base_url if settings_base_url else self.DEFAULT_BASE_URL
            self._use_azure_auth = False
        
        # Store any additional kwargs for future use
        self._extra_config = kwargs
    
    def chat(
        self,
        messages: List[Message],
        trace: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Generate text using the configured OpenAI API surface.
        
        Args:
            messages: List of conversation messages.
            trace: Optional TraceContext for observability (reserved for Stage F).
            **kwargs: Override parameters (temperature, max_tokens, etc.).
        
        Returns:
            ChatResponse with generated content and metadata.
        
        Raises:
            ValueError: If messages are invalid.
            OpenAILLMError: If API call fails.
        """
        # Validate input
        self.validate_messages(messages)
        
        # Prepare request parameters
        chat_configs = self._merge_chat_configs(kwargs)
        temperature = chat_configs.pop("temperature", self.default_temperature)
        max_tokens = chat_configs.pop("max_tokens", self.default_max_tokens)
        model = chat_configs.pop("model", self.model)
        
        # Convert messages to API format
        api_messages = [{"role": m.role, "content": m.content} for m in messages]
        
        # Make API call
        try:
            response_data = self._call_api(
                messages=api_messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_payload=chat_configs,
            )
            
            content = self._extract_content(response_data)
            usage = self._normalise_usage(response_data.get("usage"))
            
            return ChatResponse(
                content=content,
                model=response_data.get("model", model),
                usage=usage,
                raw_response=response_data,
            )
        except KeyError as e:
            raise OpenAILLMError(
                f"[OpenAI] Unexpected response format: missing key {e}"
            ) from e
        except Exception as e:
            if isinstance(e, OpenAILLMError):
                raise
            raise OpenAILLMError(
                f"[OpenAI] API call failed: {type(e).__name__}: {e}"
            ) from e
    
    def _call_api(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make the actual API call to OpenAI.
        
        This method is separated to allow easy mocking in tests.
        
        Args:
            messages: Messages in API format.
            model: Model identifier.
            temperature: Generation temperature.
            max_tokens: Maximum tokens to generate.
        
        Returns:
            Raw API response as dictionary.
        
        Raises:
            OpenAILLMError: If the API call fails.
        """
        import httpx
        
        endpoint = "chat/completions" if self.api_mode == "chat_completions" else "responses"
        url = f"{self.base_url.rstrip('/')}/{endpoint}"
        if self.api_version:
            url += f"?api-version={self.api_version}"
        
        if self._use_azure_auth:
            headers = {
                "api-key": self.api_key,
                "Content-Type": "application/json",
            }
        else:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        if self.api_mode == "chat_completions":
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        else:
            payload = {
                "model": model,
                "input": messages,
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
        if extra_payload:
            payload.update(extra_payload)
        
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, json=payload, headers=headers)
                
                if response.status_code != 200:
                    error_detail = self._parse_error_response(response)
                    raise OpenAILLMError(
                        f"[OpenAI] API error (HTTP {response.status_code}): {error_detail}"
                    )
                
                return response.json()
        except httpx.TimeoutException as e:
            raise OpenAILLMError(
                f"[OpenAI] Request timed out after 60 seconds"
            ) from e
        except httpx.RequestError as e:
            raise OpenAILLMError(
                f"[OpenAI] Connection failed: {type(e).__name__}: {e}"
            ) from e

    def _extract_content(self, response_data: Dict[str, Any]) -> str:
        """Extract assistant text from either supported response format."""
        if self.api_mode == "chat_completions":
            return response_data["choices"][0]["message"]["content"]

        output_text = response_data.get("output_text")
        if isinstance(output_text, str) and output_text:
            return output_text

        text_parts: List[str] = []
        for output_item in response_data.get("output", []):
            if not isinstance(output_item, dict) or output_item.get("type") != "message":
                continue
            for content_item in output_item.get("content", []):
                if not isinstance(content_item, dict):
                    continue
                text = content_item.get("text")
                if content_item.get("type") == "output_text" and isinstance(text, str):
                    text_parts.append(text)
        if not text_parts:
            raise KeyError("output[].content[].text")
        return "".join(text_parts)

    def _normalise_usage(
        self,
        usage: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, int]]:
        """Map Responses token names to the project's ChatResponse contract."""
        if usage is None or self.api_mode == "chat_completions":
            return usage
        return {
            "prompt_tokens": int(usage.get("input_tokens", 0)),
            "completion_tokens": int(usage.get("output_tokens", 0)),
            "total_tokens": int(usage.get("total_tokens", 0)),
        }
    
    def _parse_error_response(self, response: Any) -> str:
        """Parse error details from API response.
        
        Args:
            response: The HTTP response object.
        
        Returns:
            Human-readable error message.
        """
        try:
            error_data = response.json()
            if "error" in error_data:
                error = error_data["error"]
                if isinstance(error, dict):
                    return error.get("message", str(error))
                return str(error)
            return response.text
        except Exception:
            return response.text or "Unknown error"
