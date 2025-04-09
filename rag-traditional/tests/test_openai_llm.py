import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from src.llm.openai_llm import OpenAILLM


class TestOpenAILLM:
    @pytest.fixture
    def mock_openai(self):
        # Create a mock response
        mock_response = MagicMock(spec=ChatCompletion)
        mock_choice = MagicMock(spec=Choice)
        mock_message = MagicMock(spec=ChatCompletionMessage)
        mock_message.content = "Test response"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        # Create a mock client
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        # Create a patch for the AsyncOpenAI class and initialize method
        with patch("openai.AsyncOpenAI", return_value=mock_client):
            yield mock_client

    @pytest.mark.asyncio
    async def test_prompt_with_system_message(self, mock_openai):
        # Set environment variable for testing
        os.environ["OPENAI_API_KEY"] = "test_api_key"

        # Create a mock LLM instance with all required properties
        with patch.object(OpenAILLM, "__init__", return_value=None):
            llm = OpenAILLM()
            llm.client = mock_openai
            llm.model = "gpt-4-turbo"
            llm.api_key = "test_api_key"

        # Call prompt method
        response = await llm.prompt(
            prompt="Test prompt",
            system_message="Test system message",
            temperature=0.5,
            max_tokens=100,
        )

        # Verify response
        assert response == "Test response"

        # Verify correct parameters were passed to OpenAI client
        mock_openai.chat.completions.create.assert_called_once()
        call_args = mock_openai.chat.completions.create.call_args[1]

        assert call_args["model"] == "gpt-4-turbo"
        assert call_args["temperature"] == 0.5
        assert call_args["max_tokens"] == 100

        messages = call_args["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Test system message"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Test prompt"

    @pytest.mark.asyncio
    async def test_prompt_without_system_message(self, mock_openai):
        # Set environment variable for testing
        os.environ["OPENAI_API_KEY"] = "test_api_key"

        # Create a mock LLM instance with all required properties
        with patch.object(OpenAILLM, "__init__", return_value=None):
            llm = OpenAILLM()
            llm.client = mock_openai
            llm.model = "gpt-4-turbo"
            llm.api_key = "test_api_key"

        # Call prompt method
        response = await llm.prompt(
            prompt="Test prompt",
            temperature=0.7,
        )

        # Verify response
        assert response == "Test response"

        # Verify correct parameters were passed to OpenAI client
        mock_openai.chat.completions.create.assert_called_once()
        call_args = mock_openai.chat.completions.create.call_args[1]

        assert call_args["model"] == "gpt-4-turbo"
        assert call_args["temperature"] == 0.7

        messages = call_args["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Test prompt"

    @pytest.mark.asyncio
    async def test_prompt_with_none_response(self, mock_openai):
        # Mock a None response content
        mock_response = MagicMock(spec=ChatCompletion)
        mock_choice = MagicMock(spec=Choice)
        mock_message = MagicMock(spec=ChatCompletionMessage)
        mock_message.content = None
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        mock_openai.chat.completions.create.return_value = mock_response

        # Set environment variable for testing
        os.environ["OPENAI_API_KEY"] = "test_api_key"

        # Create a mock LLM instance with all required properties
        with patch.object(OpenAILLM, "__init__", return_value=None):
            llm = OpenAILLM()
            llm.client = mock_openai
            llm.model = "gpt-4-turbo"
            llm.api_key = "test_api_key"

        # Call prompt method
        response = await llm.prompt(prompt="Test prompt")

        # Verify response is empty string instead of None
        assert response == ""

    def test_init_with_missing_api_key(self):
        # Remove environment variable for testing
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

        # Verify ValueError is raised when no API key is provided
        with pytest.raises(ValueError):
            OpenAILLM()

    def test_init_with_custom_model(self):
        # Set environment variable for testing
        os.environ["OPENAI_API_KEY"] = "test_api_key"

        # Create OpenAILLM instance with custom model
        llm = OpenAILLM(model="gpt-3.5-turbo")

        # Verify model was set correctly
        assert llm.model == "gpt-3.5-turbo"

    @pytest.mark.asyncio
    async def test_prompt_stream(self):
        # Set environment variable for testing
        os.environ["OPENAI_API_KEY"] = "test_api_key"

        # Create mock stream chunks
        mock_chunks = [
            MagicMock(choices=[MagicMock(delta=MagicMock(content="Chunk"))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content=" 1"))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content=None))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content=" and"))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content=" 2"))]),
        ]

        # Setup mock async iterator
        mock_stream = AsyncMock()
        mock_stream.__aiter__.return_value = mock_chunks

        # Create a mock client
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream)

        # Create a mock LLM instance
        with patch.object(OpenAILLM, "__init__", return_value=None):
            llm = OpenAILLM()
            llm.client = mock_client
            llm.model = "gpt-4-turbo"
            llm.api_key = "test_api_key"

        # Call prompt_stream method
        chunks = []
        async for chunk in llm.prompt_stream(
            prompt="Test prompt",
            system_message="Test system message",
            temperature=0.5,
            max_tokens=100,
        ):
            chunks.append(chunk)

        # Verify chunks received
        assert chunks == ["Chunk", " 1", " and", " 2"]

        # Verify correct parameters were passed to OpenAI client
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args[1]

        assert call_args["model"] == "gpt-4-turbo"
        assert call_args["temperature"] == 0.5
        assert call_args["max_tokens"] == 100
        assert call_args["stream"] is True

        messages = call_args["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Test system message"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Test prompt"
