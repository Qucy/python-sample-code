"""
Azure Identity utility for authenticating to Azure OpenAI via Microsoft Entra ID.

Provides:
- ClientSecretCredential creation from environment variables.
- Token provider (bearer token provider) for the Azure Cognitive Services scope.

Environment variables required for service principal auth:
- AZURE_CLIENT_ID
- AZURE_TENANT_ID
- AZURE_CLIENT_SECRET
"""

import os
from typing import Optional

from azure.identity import ClientSecretCredential, get_bearer_token_provider

COGNITIVE_SERVICES_SCOPE = "https://cognitiveservices.azure.com/.default"


class AzureIdentityUtil:
    """Helper for Microsoft Entra ID (Azure AD) authentication.

    Usage:
        identity = AzureIdentityUtil.from_env()
        token_provider = identity.get_token_provider()
    """

    def __init__(self, credential: ClientSecretCredential):
        self.credential = credential

    @classmethod
    def from_env(
        cls,
        client_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        client_secret: Optional[str] = None,
    ) -> "AzureIdentityUtil":
        """Create a ClientSecretCredential from environment variables or provided args."""
        cid = client_id or os.getenv("AZURE_CLIENT_ID")
        tid = tenant_id or os.getenv("AZURE_TENANT_ID")
        secret = client_secret or os.getenv("AZURE_CLIENT_SECRET")

        missing = [
            name
            for name, val in {
                "AZURE_CLIENT_ID": cid,
                "AZURE_TENANT_ID": tid,
                "AZURE_CLIENT_SECRET": secret,
            }.items()
            if not val
        ]
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}"
            )

        cred = ClientSecretCredential(
            tenant_id=tid,
            client_id=cid,
            client_secret=secret,
        )
        return cls(credential=cred)

    def get_token_provider(self):
        """Return a bearer token provider for the Azure Cognitive Services scope."""
        return get_bearer_token_provider(self.credential, COGNITIVE_SERVICES_SCOPE)

    def get_token(self) -> str:
        """Fetch a raw bearer token string for the Azure Cognitive Services scope."""
        token = self.credential.get_token(COGNITIVE_SERVICES_SCOPE)
        return token.token