"""
High-level chat utility built on top of AzureOpenAIClientFactory.

Design goals from common patterns on the internet:
- Provide a simple session object that maintains message history (system/user/assistant).
- Offer a one-shot helper for quick prompts without tracking history.
- Keep dependencies light: rely on duck typing for the factory to avoid hard imports.

Usage:
    session = ChatSession(client_factory, deployment="my-deploy", system_prompt="You are helpful.")
    reply = session.send("Hello!")
    print(reply)

    text = ChatUtil(client_factory).quick_chat("my-deploy", "Hello!")
    print(text)
"""

from typing import List, Dict, Optional


class ChatSession:
    """
    Maintains a conversation history and uses a provided client factory to call
    Azure OpenAI Chat Completions. The factory only needs a `create_client()` method
    that returns an object with `chat.completions.create(...)`.
    """

    def __init__(
        self,
        client_factory,
        deployment: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
    ):
        if not deployment:
            raise ValueError("deployment (model) name is required for ChatSession.")
        self.client_factory = client_factory
        self.deployment = deployment
        self.temperature = temperature
        self._initial_system_prompt = system_prompt
        self.messages: List[Dict[str, str]] = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def send(self, user_message: str) -> str:
        """
        Add a user message, call Azure OpenAI, append the assistant reply, and return text.
        """
        self.messages.append({"role": "user", "content": user_message})
        client = self.client_factory.create_client()
        resp = client.chat.completions.create(
            model=self.deployment,
            messages=self.messages,
            temperature=self.temperature,
        )
        text = resp.choices[0].message.content
        self.messages.append({"role": "assistant", "content": text})
        return text

    def history(self) -> List[Dict[str, str]]:
        """Return the current message history (system/user/assistant)."""
        return list(self.messages)

    def reset(self) -> None:
        """Clear history and restore initial system prompt if one was set."""
        self.messages.clear()
        if self._initial_system_prompt:
            self.messages.append({"role": "system", "content": self._initial_system_prompt})


class ChatUtil:
    """
    A lightweight facade exposing one-shot chat functionality.

    This is useful for application layers where the call is stateless and no
    conversation history is needed.
    """

    def __init__(self, client_factory):
        self.client_factory = client_factory

    def quick_chat(
        self,
        deployment: str,
        user_message: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
    ) -> str:
        session = ChatSession(
            client_factory=self.client_factory,
            deployment=deployment,
            system_prompt=system_prompt,
            temperature=temperature,
        )
        return session.send(user_message)

    def batch_chat(
        self,
        deployment: str,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_workers: int = 5,
    ) -> List[str]:
        """
        Send multiple independent prompts concurrently and return replies ordered to match inputs.

        Notes:
        - This uses client-side concurrency to emulate a batch request and reduce wall-clock time.
        - Azure/OpenAI offer a server-side Batch API that is asynchronous and preview-only in Azure; this helper
          keeps things simple and compatible with GA chat completions.
        """
        if not deployment:
            raise ValueError("deployment (model) name is required for batch_chat.")
        if not prompts:
            return []

        from concurrent.futures import ThreadPoolExecutor

        def _call(prompt: str) -> str:
            session = ChatSession(
                client_factory=self.client_factory,
                deployment=deployment,
                system_prompt=system_prompt,
                temperature=temperature,
            )
            return session.send(prompt)

        results: List[str] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_call, p) for p in prompts]
            for fut in futures:
                results.append(fut.result())
        return results