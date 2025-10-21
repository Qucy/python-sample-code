import os
import unittest

from src.common.chat_util import ChatSession, ChatUtil
from src.common.azure_identity import AzureIdentityUtil
from src.common.azure_openai_factory import AzureOpenAIClientFactory

COMMON_ENV = [
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_DEPLOYMENT",
]
IDENTITY_ENV = [
    "AZURE_CLIENT_ID",
    "AZURE_TENANT_ID",
    "AZURE_CLIENT_SECRET",
]


def _missing_env(vars):
    return [v for v in vars if not os.getenv(v)]


class TestChatUtilIntegration(unittest.TestCase):
    """Real Azure integration tests for ChatSession/ChatUtil (skips without env)."""

    def test_session_with_identity(self):
        missing = _missing_env(COMMON_ENV + IDENTITY_ENV)
        if missing:
            self.skipTest("Missing env vars: " + ", ".join(missing))

        # Identity token provider
        token_provider = AzureIdentityUtil.from_env().get_token_provider()
        factory = AzureOpenAIClientFactory.from_env_with_identity(token_provider)

        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        session = ChatSession(factory, deployment=deployment, system_prompt="You are helpful.")
        reply = session.send("Hello")
        print("ChatSession (identity) reply:", reply)
        self.assertIsInstance(reply, str)
        self.assertTrue(len(reply) > 0)

    def test_quick_chat_with_api_key(self):
        missing = _missing_env(COMMON_ENV + ["AZURE_OPENAI_API_KEY"])
        if missing:
            self.skipTest("Missing env vars: " + ", ".join(missing))

        factory = AzureOpenAIClientFactory.from_env_with_api_key()
        util = ChatUtil(factory)
        text = util.quick_chat(os.getenv("AZURE_OPENAI_DEPLOYMENT"), "Hello from quick_chat")
        print("ChatUtil.quick_chat (api key) reply:", text)
        self.assertIsInstance(text, str)
        self.assertTrue(len(text) > 0)


if __name__ == "__main__":
    unittest.main()