import asyncio
import os
from typing import List, Tuple, Iterable, Union

from dotenv import load_dotenv
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI, OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),  # or the key directly (not recommended)
    # base_url="https://api.openai.com/v1",  # optional if using the default value
)


class OpenAIResponsesWrapper:
    """
    Wrapper for models exposed only via /v1/responsescodex
    (e.g., gpt-5-codex). Provides an asynchronous ainvoke() method
    that returns an AIMessage similar to LangChain models.
    """

    def __init__(self, client, model):
        self.client = client
        self.model = model

    def __call__(self, prompt):
        resp = self.client.responses.create(
            model=self.model,
            input=prompt,
        )
        return resp.output_text  # or the structure returned by your endpoint

    def _to_openai_messages(self, messages: List[BaseMessage]):
        """
        Converts LangChain messages to the format required by /responses.
        """
        converted = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "user"
                content_type = "input_text"
            elif isinstance(msg, SystemMessage):
                role = "system"
                content_type = "input_text"
            elif isinstance(msg, AIMessage):
                role = "assistant"
                content_type = "output_text"
            else:
                role = getattr(msg, "role", "user")
                content_type = "input_text"

            converted.append(
                {
                    "role": role,
                    "content": [
                        {
                            "type": content_type,
                            "text": msg.content,
                        }
                    ],
                }
            )
        return converted

    @staticmethod
    def _iter_content(item: Union[dict, object]) -> Iterable[dict]:
        """
        Extracts the 'content' list from a response.output item,
        supporting both dicts and objects.
        """
        if isinstance(item, dict):
            return item.get("content", []) or []
        return getattr(item, "content", None) or []

    @staticmethod
    def _extract_text_block(block: Union[str, dict, object]) -> str:
        """
        Extracts the text portion from an output_text block.
        """
        if isinstance(block, str):
            return block
        if isinstance(block, dict):
            return block.get("value") or block.get("text") or ""
        if hasattr(block, "value"):
            return block.value
        if hasattr(block, "text"):
            return str(block.text)
        return ""

    def _extract_text(self, response) -> str:
        """
        Extracts text from the response returned by /responses.
        Supports various output structures.
        """
        chunks = []

        output_items = getattr(response, "output", None) or []
        for item in output_items:
            for content in self._iter_content(item):
                if getattr(content, "type", None) == "output_text" or (
                    isinstance(content, dict) and content.get("type") == "output_text"
                ):
                    text_block = getattr(content, "text", None)
                    if text_block is None and isinstance(content, dict):
                        text_block = content.get("text")
                    extracted = self._extract_text_block(text_block)
                    if extracted:
                        chunks.append(extracted)

        if chunks:
            return "".join(chunks)

        fallback_text = getattr(response, "output_text", None)
        if fallback_text:
            return fallback_text

        content_items = getattr(response, "content", None) or []
        for content in content_items:
            if hasattr(content, "text"):
                extracted = self._extract_text_block(content.text)
                if extracted:
                    return extracted
            if isinstance(content, dict):
                extracted = content.get("text") or content.get("value", "")
                if extracted:
                    return extracted

        return ""

    async def ainvoke(self, messages: List[BaseMessage]):
        payload = {
            "model": self.model,
            "input": self._to_openai_messages(messages),
        }
        payload.update(self.kwargs)

        response = await self.client.responses.create(**payload)
        text = self._extract_text(response)
        return AIMessage(content=text)


def get_model(model_name: str):
    """
    Initializes and returns the requested model.
    """
    name = model_name.lower()

    if name == "deepseek":
        return ChatDeepSeek(
            model="deepseek-chat",
            temperature=0.2,
            max_tokens=8000,
            timeout=90,
            max_retries=2,
        )

    if name in ("gpt-4o", "gpt4o"):
        return ChatOpenAI(
            model="gpt-4o",
            temperature=0.2,
            max_tokens=5000,
            timeout=90,
            max_retries=2,
        )

    if name == "gpt5-codex":
        return ChatOpenAI(
            model="gpt-5",
            temperature=0.2,
            max_tokens=5000,
        )

    raise ValueError("Unsupported model. Choose from 'Deepseek', 'gpt-4o', or 'gpt-5-codex'.")


async def llm_models():
    """
    Loads environment variables and tests each model asynchronously.
    """
    load_dotenv()
    print("✅ Environment variables loaded (if a .env file exists).")
    print("-" * 50)

    models: List[Tuple[str, object]] = []

    for name in ("Deepseek", "gpt-4o", "gpt-5-codex"):
        try:
            model = get_model(name)
            models.append((name, model))
        except Exception as exc:
            print(f"⚠️  Could not initialize '{name}': {exc}")

    if not models:
        print("\n⛔ TEST FAILED: No models could be initialized.")
        print("   Check your keys in the .env file.")
        return

    print(f"\n✅ Models ready for testing: {', '.join(name for name, _ in models)}")
    print("-" * 50)

    test_message = [HumanMessage(content="Describe your main purpose in a short sentence.")]

    tasks = []
    for name, model in models:
        tasks.append((name, model.ainvoke(test_message)))

    for name, task in tasks:
        try:
            response = await task
            text = (response.content or "").strip()
            summary = text[:80] + ("..." if len(text) > 80 else "")
            print(f"🤖 {name} (OK):\n   Response: '{summary}'\n")
        except Exception as exc:
            print(f"❌ {name} (FAILED):\n   Execution error: {exc}\n")

    print("-" * 50)
    print("🏁 Model testing completed.")


def main():
    asyncio.run(llm_models())
