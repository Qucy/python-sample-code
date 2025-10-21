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

    def test_session_multi_turn_with_identity(self):
        missing = _missing_env(COMMON_ENV + IDENTITY_ENV)
        if missing:
            self.skipTest("Missing env vars: " + ", ".join(missing))

        token_provider = AzureIdentityUtil.from_env().get_token_provider()
        factory = AzureOpenAIClientFactory.from_env_with_identity(token_provider)
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

        session = ChatSession(factory, deployment=deployment, system_prompt="You are helpful.")
        reply1 = session.send("Hello")
        print("ChatSession (identity) turn1:", reply1)
        self.assertIsInstance(reply1, str)
        self.assertTrue(len(reply1) > 0)

        reply2 = session.send("What is 2 + 2?")
        print("ChatSession (identity) turn2:", reply2)
        self.assertIsInstance(reply2, str)
        self.assertTrue(len(reply2) > 0)

        # Expect at least system + 2 user + 2 assistant messages
        self.assertTrue(len(session.history()) >= 5)

    def test_quick_chat_with_identity(self):
        missing = _missing_env(COMMON_ENV + IDENTITY_ENV)
        if missing:
            self.skipTest("Missing env vars: " + ", ".join(missing))

        token_provider = AzureIdentityUtil.from_env().get_token_provider()
        factory = AzureOpenAIClientFactory.from_env_with_identity(token_provider)
        text = ChatUtil(factory).quick_chat(os.getenv("AZURE_OPENAI_DEPLOYMENT"), "Hello from identity quick_chat")
        print("ChatUtil.quick_chat (identity) reply:", text)
        self.assertIsInstance(text, str)
        self.assertTrue(len(text) > 0)

    def test_quick_chat_with_api_key_system_prompt(self):
        missing = _missing_env(COMMON_ENV + ["AZURE_OPENAI_API_KEY"])
        if missing:
            self.skipTest("Missing env vars: " + ", ".join(missing))

        factory = AzureOpenAIClientFactory.from_env_with_api_key()
        text = ChatUtil(factory).quick_chat(
            os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            "Explain Azure OpenAI briefly.",
            system_prompt="Answer very briefly.",
        )
        print("ChatUtil.quick_chat (api key + system prompt) reply:", text)
        self.assertIsInstance(text, str)
        self.assertTrue(len(text) > 0)

    def test_session_reset_behavior_with_identity(self):
        missing = _missing_env(COMMON_ENV + IDENTITY_ENV)
        if missing:
            self.skipTest("Missing env vars: " + ", ".join(missing))

        token_provider = AzureIdentityUtil.from_env().get_token_provider()
        factory = AzureOpenAIClientFactory.from_env_with_identity(token_provider)
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

        session = ChatSession(factory, deployment=deployment, system_prompt="You are helpful.")
        _ = session.send("First turn")
        _ = session.send("Second turn")
        self.assertTrue(len(session.history()) >= 5)

        session.reset()
        self.assertTrue(len(session.history()) == 1)
        reply = session.send("New conversation: say hi.")
        print("ChatSession (identity) after reset:", reply)
        self.assertIsInstance(reply, str)
        self.assertTrue(len(reply) > 0)


if __name__ == "__main__":
    unittest.main()