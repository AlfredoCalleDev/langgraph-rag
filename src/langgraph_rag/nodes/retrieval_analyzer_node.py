from langchain_core.runnables import Runnable
from langgraph_rag.states.RAGState import RAGState


def retrieval_analyzer_node(state: RAGState, analyzer_chain: Runnable) -> dict:
    """
    Analiza si se necesita realizar retrieval.
    Args:
        state (RAGState): Estado global del RAG.
        analyzer_chain (Runnable): Chain para analizar.
    Returns:
        dict: Estado actualizado.
    """
    question = state["question"]
    response = analyzer_chain.invoke({"question": question})

    # print(f"\n 🧐 [retrieval_analyzer_node] response: {response}")

    return {
        "retrieval_analyzer_response": response,
    }
