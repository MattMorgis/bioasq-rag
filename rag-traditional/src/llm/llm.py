from abc import ABC, abstractmethod
from typing import Optional


class LLM(ABC):
    """
    Abstract base class for language model interfaces.
    Implementations should handle specific LLM providers like OpenAI, Claude, etc.
    """

    @abstractmethod
    async def prompt(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Send a prompt to the language model and get a response.

        Args:
            prompt: The user prompt to send to the model
            system_message: Optional system message to set context
            temperature: Controls randomness of output (0.0-1.0)
            max_tokens: Maximum number of tokens to generate

        Returns:
            The model's response as a string
        """
        pass
