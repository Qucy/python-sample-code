import os
import unittest

from src.common.azure_identity import AzureIdentityUtil
from src.common.azure_openai_factory import AzureOpenAIClientFactory

REQUIRED_ENV_VARS = [
    "AZURE_CLIENT_ID",
    "AZURE_TENANT_ID",
    "AZURE_CLIENT_SECRET",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_DEPLOYMENT",
]


class TestAzureIdentityIntegration(unittest.TestCase):
    """Integration tests that exercise real Entra ID auth and Azure OpenAI.

    These tests will be skipped unless all required environment variables are present.
    """

    def setUp(self):
        missing = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
        if missing:
            self.skipTest(
                "Skipping real Azure integration test; missing env vars: " + ", ".join(missing)
            )

    def test_aad_token_and_chat_completion(self):
        # 1) Identity from environment (service principal)
        identity = AzureIdentityUtil.from_env()

        # 2) Ensure we can fetch a real bearer token
        token = identity.get_token()
        self.assertIsInstance(token, str)
        self.assertTrue(len(token) > 0)

        # 3) Create AzureOpenAI client via token provider
        token_provider = identity.get_token_provider()
        factory = AzureOpenAIClientFactory.from_env_with_identity(token_provider)

        # 4) Perform a minimal chat completions call
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        reply = factory.quick_chat(deployment, "ping")
        self.assertIsInstance(reply, str)
        self.assertTrue(len(reply) > 0)


if __name__ == "__main__":
    unittest.main()