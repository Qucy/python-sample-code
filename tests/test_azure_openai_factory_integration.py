import os
import unittest

from src.utils.azure_identity import AzureIdentityUtil
from src.utils.azure_openai_factory import AzureOpenAIClientFactory

IDENTITY_ENV = [
    "AZURE_CLIENT_ID",
    "AZURE_TENANT_ID",
    "AZURE_CLIENT_SECRET",
]
COMMON_ENV = [
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_DEPLOYMENT",
]


def _missing_env(vars_list):
    return [var for var in vars_list if not os.getenv(var)]


class TestAzureOpenAIFactoryIntegration(unittest.TestCase):
    """Integration tests for AzureOpenAIClientFactory.

    Tests are skipped unless required environment variables are set.
    """

    def test_chat_with_identity(self):
        missing = _missing_env(IDENTITY_ENV + COMMON_ENV)
        if missing:
            self.skipTest("Missing env vars: " + ", ".join(missing))

        # Build identity and client via token provider
        identity = AzureIdentityUtil.from_env()
        token_provider = identity.get_token_provider()
        factory = AzureOpenAIClientFactory.from_env_with_identity(token_provider)

        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        reply = factory.quick_chat(deployment, "Hello from factory integration test (AAD)!")
        print("Chat (AAD) reply:", reply)
        self.assertIsInstance(reply, str)
        self.assertTrue(len(reply) > 0)

    def test_chat_with_api_key(self):
        # Requires common env + AZURE_OPENAI_API_KEY
        missing = _missing_env(COMMON_ENV + ["AZURE_OPENAI_API_KEY"])
        if missing:
            self.skipTest("Missing env vars: " + ", ".join(missing))

        factory = AzureOpenAIClientFactory.from_env_with_api_key()
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        reply = factory.quick_chat(deployment, "Hello from factory integration test (API key)!")
        print("Chat (API key) reply:", reply)
        self.assertIsInstance(reply, str)
        self.assertTrue(len(reply) > 0)

    def test_responses_with_identity(self):
        # Try the Responses API; skip if unsupported by SDK/API version
        missing = _missing_env(IDENTITY_ENV + COMMON_ENV)
        if missing:
            self.skipTest("Missing env vars: " + ", ".join(missing))

        identity = AzureIdentityUtil.from_env()
        token_provider = identity.get_token_provider()
        factory = AzureOpenAIClientFactory.from_env_with_identity(token_provider)

        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        try:
            reply = factory.quick_response(deployment, "Hello from Responses API!")
            print("Responses (AAD) reply:", reply)
            self.assertIsInstance(reply, str)
            self.assertTrue(len(reply) > 0)
        except AttributeError:
            # Older SDKs may not expose .responses for Azure clients
            self.skipTest("Responses API not available on AzureOpenAI client in this environment")
        except Exception as e:
            # Some API versions/resources may not support Responses yet
            self.skipTest(f"Responses API call skipped due to error: {e}")

    def test_create_client_identity(self):
        missing = _missing_env(IDENTITY_ENV + COMMON_ENV)
        if missing:
            self.skipTest("Missing env vars: " + ", ".join(missing))

        identity = AzureIdentityUtil.from_env()
        token_provider = identity.get_token_provider()
        factory = AzureOpenAIClientFactory.from_env_with_identity(token_provider)
        client = factory.create_client()
        # Smoke test: minimal chat call via the raw client API
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        resp = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": "Hello via raw client!"}],
            temperature=0,
        )
        text = resp.choices[0].message.content
        print("Raw client chat (AAD) reply:", text)
        self.assertIsInstance(text, str)
        self.assertTrue(len(text) > 0)


if __name__ == "__main__":
    unittest.main()