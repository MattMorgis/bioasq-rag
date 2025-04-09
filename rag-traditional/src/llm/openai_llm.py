import os
from typing import List, Optional

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from src.llm.llm import LLM


class OpenAILLM(LLM):
    """
    OpenAI implementation of the LLM interface.
    Handles communication with OpenAI API for language model interactions.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4-turbo",
    ):
        """
        Initialize the OpenAI LLM client.

        Args:
            api_key: OpenAI API key. If None, will attempt to get from OPENAI_API_KEY env var
            model: The model to use (default: gpt-4-turbo)
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key must be provided either as an argument or through the OPENAI_API_KEY environment variable"
            )

        self.model = model
        self.client = AsyncOpenAI(api_key=self.api_key)

    async def prompt(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Send a prompt to the OpenAI language model and get a response.

        Args:
            prompt: The user prompt to send to the model
            system_message: Optional system message to set context
            temperature: Controls randomness of output (0.0-1.0)
            max_tokens: Maximum number of tokens to generate

        Returns:
            The model's response as a string
        """
        messages: List[ChatCompletionMessageParam] = []

        if system_message:
            system_msg: ChatCompletionSystemMessageParam = {
                "role": "system",
                "content": system_message,
            }
            messages.append(system_msg)

        user_msg: ChatCompletionUserMessageParam = {"role": "user", "content": prompt}
        messages.append(user_msg)

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        content = response.choices[0].message.content
        return content if content is not None else ""
