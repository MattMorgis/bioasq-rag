import os
import uuid
from datetime import datetime

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from .llm.openai_llm import OpenAILLM
from .models import RAGRequest, RAGResponse
from .prompt import (
    create_rag_prompt,
    get_rag_system_message,
)
from .search.vector_client import VectorSearchClient
from .utils import clean_title, format_context_from_results

# Load environment variables
load_dotenv()

# Create prompts directory if it doesn't exist
PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")
os.makedirs(PROMPTS_DIR, exist_ok=True)

router = APIRouter()


def save_prompt_to_file(query: str, prompt: str, system_message: str) -> str:
    """
    Save the prompt and system message to a file.

    Args:
        query: The user's query
        prompt: The formatted RAG prompt
        system_message: The system message for the LLM

    Returns:
        Path to the saved prompt file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    filename = f"prompt_{timestamp}_{unique_id}.txt"
    filepath = os.path.join(PROMPTS_DIR, filename)

    with open(filepath, "w") as f:
        f.write(f"TIMESTAMP: {datetime.now().isoformat()}\n")
        f.write(f"QUERY: {query}\n\n")
        f.write("SYSTEM MESSAGE:\n")
        f.write(system_message)
        f.write("\n\nPROMPT:\n")
        f.write(prompt)

    return filepath


@router.post("/rag/query", response_model=RAGResponse)
async def query_rag(request: RAGRequest) -> RAGResponse:
    """
    RAG endpoint that takes a biomedical query, retrieves relevant abstracts,
    and generates an evidence-based answer.

    Args:
        request: RAGRequest containing the query and optional parameters

    Returns:
        RAGResponse with the generated answer and source information
    """
    try:
        # Step 1: Perform vector search to find relevant documents
        search_client = VectorSearchClient()
        search_response = await search_client.search(
            query=request.query, limit=request.max_results
        )

        # Step 2: Format search results as context for the LLM
        context = format_context_from_results(search_response.results)

        # Step 3: Create a prompt for the LLM using the prompt module
        prompt = create_rag_prompt(query=request.query, context=context)
        system_message = get_rag_system_message()

        # Save prompt to file if logging is enabled
        if hasattr(request, "log_prompt") and request.log_prompt:
            save_prompt_to_file(request.query, prompt, system_message)

        # Step 4: Generate an answer using the LLM
        llm = OpenAILLM()
        answer = await llm.prompt(
            prompt=prompt,
            system_message=system_message,
            temperature=request.temperature,
        )

        # Step 5: Format sources for the response
        sources = [
            {
                "title": clean_title(result.title),
                "chunk": result.text,
                "abstract_id": result.abstract_id,
                "url": result.url,
                "score": result.score,
                "publication_date": result.publication_date,
                "journal": result.journal,
                "authors": result.authors,
            }
            for result in search_response.results
        ]

        # Step 6: Return the RAG response
        return RAGResponse(query=request.query, answer=answer, sources=sources)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing RAG request: {str(e)}"
        )


@router.post("/rag/query/stream")
async def query_rag_stream(request: RAGRequest):
    """
    Streaming RAG endpoint that takes a biomedical query, retrieves relevant abstracts,
    and generates an evidence-based answer with the response streamed back to the client as plain text.

    Args:
        request: RAGRequest containing the query and optional parameters

    Returns:
        StreamingResponse with the generated answer as plain text chunks, followed by source information
    """
    try:
        # Step 1: Perform vector search to find relevant documents
        search_client = VectorSearchClient()
        search_response = await search_client.search(
            query=request.query, limit=request.max_results
        )

        # Step 2: Format search results as context for the LLM
        context = format_context_from_results(search_response.results)

        # Step 3: Create a prompt for the LLM using the prompt module
        prompt = create_rag_prompt(query=request.query, context=context)
        system_message = get_rag_system_message()

        # Save prompt to file if logging is enabled
        if hasattr(request, "log_prompt") and request.log_prompt:
            save_prompt_to_file(request.query, prompt, system_message)

        # Step 4: Create the streaming generator function
        async def stream_generator():
            # Stream the LLM response as plain text
            llm = OpenAILLM()
            async for chunk in llm.prompt_stream(
                prompt=prompt,
                system_message=system_message,
                temperature=request.temperature,
            ):
                yield chunk

            # Append sources information in plain text at the end
            yield "\n\n--- Sources ---\n"

            for i, result in enumerate(search_response.results, 1):
                title = clean_title(result.title or "Unknown Journal")
                source_info = f"{i}. {title}"
                if result.publication_date:
                    source_info += f" ({result.publication_date})"
                yield f"{source_info}\n"

        # Return a streaming response with plain text
        return StreamingResponse(
            stream_generator(),
            media_type="text/plain",
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing streaming RAG request: {str(e)}"
        )
