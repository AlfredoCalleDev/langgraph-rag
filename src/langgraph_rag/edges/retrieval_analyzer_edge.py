from typing import Literal
from langgraph_rag.states.RAGState import RAGState


def retrieval_analyzer_edge(state: RAGState) -> Literal["retriever", "generator"]:
    """
    Decide si se necesita realizar retrieval o no.
    Args:
        state (RAGState): Estado global del RAG.
    Returns:
        Literal["retriever", "generator"]: Nodo a ejecutar a continuación.
    """
    retrieval_analyzer_response = state["retrieval_analyzer_response"]

    if retrieval_analyzer_response.need_retrieval:
        return "retriever"
    else:
        return "generator"
