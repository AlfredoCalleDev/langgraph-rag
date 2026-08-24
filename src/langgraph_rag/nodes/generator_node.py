from langchain_core.runnables import Runnable
from langchain.messages import AIMessage
from langgraph_rag.states.RAGState import RAGState
from langgraph_rag.utils.formatters import format_docs


def generator_node(state: RAGState, generator_chain: Runnable) -> dict:
    """Genera la respuesta

    Args:
        state (RAGState): Estado global del RAG
        generator_chain (Runnable): Chain para generar la respuesta

    Returns:
        dict: Diccionario con la respuesta generada
    """

    question = state["question"]
    retrieved_documents = state["retrieved_documents"]

    response = generator_chain.invoke(
        {"context": format_docs(retrieved_documents), "question": question}
    )

    return {
        "response": response,
        "messages": [AIMessage(content=response)],
    }
