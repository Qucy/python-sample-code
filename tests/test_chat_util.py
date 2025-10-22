import unittest

from src.common.chat_util import ChatSession, ChatUtil


class FakeMessage:
    def __init__(self, content: str):
        self.content = content


class FakeChoice:
    def __init__(self, message: FakeMessage):
        self.message = message


class FakeResponse:
    def __init__(self, text: str):
        self.choices = [FakeChoice(FakeMessage(text))]


class FakeChatAPI:
    def __init__(self, reply_text: str):
        self._reply_text = reply_text

    class completions:
        @staticmethod
        def create(model: str, messages, temperature: float):
            # Inspect input if needed; return deterministic reply
            return FakeResponse(text=f"echo: {messages[-1]['content']}")


class FakeClient:
    def __init__(self, reply_text: str = "ok"):
        self.chat = FakeChatAPI(reply_text)


class FakeFactory:
    def __init__(self, reply_text: str = "ok"):
        self._reply_text = reply_text

    def create_client(self):
        return FakeClient(reply_text=self._reply_text)


class TestChatUtil(unittest.TestCase):
    def test_chat_session_with_system_prompt_and_history(self):
        factory = FakeFactory()
        session = ChatSession(factory, deployment="demo", system_prompt="You are concise.")

        # Initial history contains the system prompt
        hist = session.history()
        self.assertEqual(len(hist), 1)
        self.assertEqual(hist[0]["role"], "system")

        # Send a user message
        reply = session.send("Hello")
        self.assertTrue(isinstance(reply, str))
        self.assertTrue(len(reply) > 0)

        # History should now contain system, user, assistant
        hist = session.history()
        self.assertEqual(len(hist), 3)
        self.assertEqual(hist[1]["role"], "user")
        self.assertEqual(hist[2]["role"], "assistant")

        # Reset should restore only the system prompt
        session.reset()
        hist = session.history()
        self.assertEqual(len(hist), 1)
        self.assertEqual(hist[0]["role"], "system")

    def test_quick_chat_stateless(self):
        factory = FakeFactory()
        util = ChatUtil(factory)
        text1 = util.quick_chat("demo", "Hello")
        text2 = util.quick_chat("demo", "How are you?")
        self.assertNotEqual(text1, "")
        self.assertNotEqual(text2, "")
        # Different prompts yield different echoes in our fake setup
        self.assertNotEqual(text1, text2)

    def test_missing_deployment_raises(self):
        factory = FakeFactory()
        with self.assertRaises(ValueError):
            ChatSession(factory, deployment="", system_prompt=None)

    def test_parse_json_text_direct(self):
        data = ChatUtil.parse_json_text('{"a": 1, "b": "x"}')
        self.assertIsInstance(data, dict)
        self.assertEqual(data["a"], 1)
        self.assertEqual(data["b"], "x")

    def test_parse_json_text_fenced(self):
        s = """
        ```json
        {"topic": "Azure", "summary": "Cloud platform by Microsoft."}
        ```
        """
        data = ChatUtil.parse_json_text(s)
        self.assertIn("topic", data)
        self.assertIn("summary", data)

    def test_parse_json_text_with_prose(self):
        s = "Here is your result: {\"k\": 2, \"v\": [1,2,3]} Enjoy!"
        data = ChatUtil.parse_json_text(s)
        self.assertEqual(data["k"], 2)
        self.assertEqual(data["v"], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()