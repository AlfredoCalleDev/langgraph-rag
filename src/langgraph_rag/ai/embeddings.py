from functools import lru_cache
from langchain_openai import OpenAIEmbeddings
from langgraph_rag.config.settings import settings


@lru_cache(maxsize=1)
def get_embedding() -> OpenAIEmbeddings:
    """Obtiene una instancia del Embedding

    Returns:
        OpenAIEmbeddings: Instancia del Embedding
    """

    return OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        max_retries=settings.MAX_RETRIES,
    )
