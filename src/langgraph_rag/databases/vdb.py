import os
from langchain_chroma import Chroma
from langgraph_rag.config.settings import settings
from langgraph_rag.ai.embeddings import get_embedding


def get_or_create_vector_store(
    collection_name: str, persist_path: str = None
) -> Chroma:
    """
    Obtiene un vector store existente o crea uno nuevo

    Args:
        collection_name (str): Nombre de la colección
        persist_path (str, optional): Ruta de persistencia

    Returns:
        Chroma: Almacén de vectores
    """

    path = persist_path or settings.CHROMA_PATH
    os.makedirs(path, exist_ok=True)

    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embedding(),
        persist_directory=path,
    )
