from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException

from .llm.openai_llm import OpenAILLM
from .models import RAGRequest, RAGResponse
from .search.vector_client import VectorSearchClient
from .utils import format_context_from_results

# Load environment variables
load_dotenv()

router = APIRouter()


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

        # Step 3: Create a prompt for the LLM
        prompt = f"""
        Answer the following biomedical question based on the provided research abstracts.
        Your answer should be accurate, concise, and based solely on the information provided.
        If the abstracts don't contain enough information to answer confidently, acknowledge the limitations.

        QUESTION: {request.query}

        ABSTRACTS:
        {context}
        """

        system_message = """
        You are a biomedical research assistant specialized in answering questions based on scientific literature.
        Provide accurate, evidence-based answers citing the sources you used.
        Only use information provided in the abstracts to form your answer.
        """

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
                "title": result.title,
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
