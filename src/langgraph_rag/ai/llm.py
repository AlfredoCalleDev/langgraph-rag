from functools import lru_cache
from langchain_openai import ChatOpenAI
from langgraph_rag.config.settings import settings


@lru_cache(maxsize=1)
def get_llm(temperature: float = None) -> ChatOpenAI:
    """Obtiene una instancia del LLM

    Args:
        temperature (float, optional): Temperatura del modelo.

    Returns:
        ChatOpenAI: Instancia del LLM
    """

    return ChatOpenAI(
        model=settings.LLM_MODEL,
        temperature=temperature or settings.DEFAULT_TEMPERATURE,
        max_retries=settings.MAX_RETRIES,
    )
