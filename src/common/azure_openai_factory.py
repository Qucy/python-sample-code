"""
Factory for creating an AzureOpenAI client using either:
- Microsoft Entra ID (Azure AD) token provider via azure-identity
- API key authentication

This uses the official OpenAI Python library's AzureOpenAI client.

Required env vars:
- AZURE_OPENAI_ENDPOINT: e.g. https://<your-resource>.openai.azure.com/
- AZURE_OPENAI_API_VERSION: e.g. 2024-10-21 (latest GA as of 2025)
Optional (for API key auth):
- AZURE_OPENAI_API_KEY
"""

import os
from typing import Optional

from openai import AzureOpenAI
import warnings


class AzureOpenAIClientFactory:
    """Create a configured AzureOpenAI client.

    Example (AAD auth):
        from src.common.azure_identity import AzureIdentityUtil
         token_provider = AzureIdentityUtil.from_env().get_token_provider()
         factory = AzureOpenAIClientFactory.from_env_with_identity(token_provider)
         client = factory.create_client()

    Example (API key auth):
        factory = AzureOpenAIClientFactory.from_env_with_api_key()
        client = factory.create_client()
    """

    def __init__(
        self,
        endpoint: str,
        api_version: str,
        azure_ad_token_provider=None,
        api_key: Optional[str] = None,
    ):
        if not endpoint:
            raise ValueError(
                "Azure OpenAI endpoint is required (set AZURE_OPENAI_ENDPOINT)."
            )
        self.endpoint = endpoint.rstrip("/")
        self.api_version = api_version
        self.azure_ad_token_provider = azure_ad_token_provider
        self.api_key = api_key

    def create_client(self) -> AzureOpenAI:
        """Instantiate and return the AzureOpenAI client with the configured auth."""
        kwargs = {
            "azure_endpoint": self.endpoint,
            "api_version": self.api_version,
        }
        if self.azure_ad_token_provider:
            kwargs["azure_ad_token_provider"] = self.azure_ad_token_provider
        elif self.api_key:
            kwargs["api_key"] = self.api_key
        else:
            raise ValueError("Provide either azure_ad_token_provider or api_key.")
        return AzureOpenAI(**kwargs)

    @classmethod
    def from_env_with_identity(
        cls,
        token_provider,
        endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
    ) -> "AzureOpenAIClientFactory":
        ep = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        ver = api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
        if not ep:
            raise ValueError("AZURE_OPENAI_ENDPOINT must be set in environment.")
        return cls(endpoint=ep, api_version=ver, azure_ad_token_provider=token_provider)

    @classmethod
    def from_env_with_api_key(
        cls,
        endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> "AzureOpenAIClientFactory":
        ep = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        ver = api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
        key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        if not ep:
            raise ValueError("AZURE_OPENAI_ENDPOINT must be set in environment.")
        if not key:
            raise ValueError("AZURE_OPENAI_API_KEY is required for API key auth.")
        return cls(endpoint=ep, api_version=ver, api_key=key)

    def quick_chat(self, deployment: str, user_message: str) -> str:
        """Deprecated: Use ChatUtil.quick_chat(...) or ChatSession."""
        warnings.warn(
            "AzureOpenAIClientFactory.quick_chat is deprecated; use ChatUtil.quick_chat or ChatSession.",
            DeprecationWarning,
        )
        client = self.create_client()
        resp = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": user_message}],
            temperature=1.0,
        )
        return resp.choices[0].message.content

    def quick_response(self, model: str, input_text: str) -> str:
        """Deprecated: Prefer ChatUtil/ChatSession or call client.responses.create directly."""
        warnings.warn(
            "AzureOpenAIClientFactory.quick_response is deprecated; prefer ChatUtil/ChatSession or client.responses.create.",
            DeprecationWarning,
        )
        client = self.create_client()
        resp = client.responses.create(model=model, input=input_text)
        if hasattr(resp, "output_text"):
            return resp.output_text
        try:
            if hasattr(resp, "output") and resp.output:
                content = resp.output[0].content if hasattr(resp.output[0], "content") else None
                if content:
                    texts = []
                    for chunk in content:
                        if getattr(chunk, "type", None) == "output_text" and hasattr(chunk, "text"):
                            texts.append(chunk.text)
                    if texts:
                        return "\n".join(texts)
        except Exception:
            pass
        return str(resp)