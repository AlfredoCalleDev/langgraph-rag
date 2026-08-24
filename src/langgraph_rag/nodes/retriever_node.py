from langchain_core.vectorstores import VectorStore
from langgraph_rag.states.RAGState import RAGState
from langgraph_rag.config.settings import settings


def retriever_node(state: RAGState, vector_store: VectorStore) -> dict:
    """Recupera los documentos (chunks) más relevantes en el vector store

    Args:
        state (RAGState): Estado global del RAG
        vector_store (VectorStore): Vector store

    Returns:
        dict: Diccionario con los documentos recuperados
    """

    retrieved_documents = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": settings.TOP_K_RESULTS}
    ).invoke(state["question"])

    return {"retrieved_documents": retrieved_documents}
