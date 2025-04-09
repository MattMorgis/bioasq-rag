from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from .llm.openai_llm import OpenAILLM
from .models import RAGRequest, RAGResponse
from .search.vector_client import VectorSearchClient
from .utils import clean_title, format_context_from_results

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
        ## Context (Retrieved Information):
        {context}

        ## Few Shot Examples:
        Example 1:
        User Query: Which receptor is inhibited by Teprotumumab?
        Assistant Response: Teprotumumab is a monoclonal inhibitory antibody targeting IGF-1 receptor.

        Example 2:
        User Query: Does the protein mTOR regulate autophagy?
        Assistant Response: mammalian target of rapamycin (mTOR)  is a major negative regulator of autophagy.

        Example 3:
        User Query: Which disease was studied in the CADISS trial?
        Assistant Response: CADISS was a prospective multicentre randomised-controlled trial in acute (within 7 days of onset) carotid and vertebral artery dissection.

        Example 4:
        User Query: Is Daprodustat effective for anemia?
        Assistant Response: Yes. Daprodustat is a hypoxia-inducible factor-prolyl hydroxylase inhibitor for the treatment of anemia of chronic kidney disease.

        Instructions for use:
        Answer the following biomedical question based on the provided research abstracts.
        Your answer should be accurate, concise, and based solely on the information provided.
        If the abstracts don't contain enough information to answer confidently, acknowledge the limitations.

        QUESTION: {request.query}

        """

        system_message = """
        You are a biomedical research assistant specialized in answering questions based on scientific literature.
        Provide accurate, evidence-based answers citing the sources you used.
        Only use information provided in the abstracts to form your answer.
        When the retrieved information doesn't cover the query, acknowledge the limitations and respond with "I'm sorry, I don't have information about that topic."
        IMPORTANT: Do not include source citations or references (like "Source 2" or "according to Document 3") in your responses. Integrate the information naturally as if you already know it. Your answers should be seamless and conversational.

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

        # Step 3: Create a prompt for the LLM
        prompt = f"""
        ## Context (Retrieved Information):
        {context}

        ## Few Shot Examples:
        Example 1:
        User Query: Which receptor is inhibited by Teprotumumab?
        Assistant Response: Teprotumumab is a monoclonal inhibitory antibody targeting IGF-1 receptor.

        Example 2:
        User Query: Does the protein mTOR regulate autophagy?
        Assistant Response: mammalian target of rapamycin (mTOR)  is a major negative regulator of autophagy.

        Example 3:
        User Query: Which disease was studied in the CADISS trial?
        Assistant Response: CADISS was a prospective multicentre randomised-controlled trial in acute (within 7 days of onset) carotid and vertebral artery dissection.

        Example 4:
        User Query: Is Daprodustat effective for anemia?
        Assistant Response: Yes. Daprodustat is a hypoxia-inducible factor-prolyl hydroxylase inhibitor for the treatment of anemia of chronic kidney disease.

        Instructions for use:
        Answer the following biomedical question based on the provided research abstracts.
        Your answer should be accurate, concise, and based solely on the information provided.
        If the abstracts don't contain enough information to answer confidently, acknowledge the limitations.

        QUESTION: {request.query}
        """

        system_message = """
        You are a biomedical research assistant specialized in answering questions based on scientific literature.
        Provide accurate, evidence-based answers citing the sources you used.
        Only use information provided in the abstracts to form your answer.
        When the retrieved information doesn't cover the query, acknowledge the limitations and respond with "I'm sorry, I don't have information about that topic."
        """

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
