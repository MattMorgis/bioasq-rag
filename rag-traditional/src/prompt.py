def create_rag_prompt(query: str, context: str) -> str:
    """
    Creates a prompt for the RAG system with the query and context

    Args:
        query: The user's biomedical query
        context: The formatted context from retrieved documents

    Returns:
        A formatted prompt string for the LLM
    """
    return f"""
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

    QUESTION: {query}
    """


def get_rag_system_message() -> str:
    """
    Returns the system message for the RAG LLM

    Returns:
        The system message string for the RAG LLM
    """
    return """
    You are a biomedical research assistant specialized in answering questions based on scientific literature.
    Provide accurate, evidence-based answers citing the sources you used.
    Only use information provided in the abstracts to form your answer.
    When the retrieved information doesn't cover the query, acknowledge the limitations and respond with "I'm sorry, I don't have information about that topic."
    IMPORTANT: Do not include source citations or references (like "Source 2" or "according to Document 3") in your responses. Integrate the information naturally as if you already know it. Your answers should be seamless and conversational.
    """
