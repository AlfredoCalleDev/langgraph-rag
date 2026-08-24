from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langgraph_rag.schemas.retrieval_analyzer_response import RetrievalAnalyzerResponse


def build_retrieval_analyzer_chain(
    prompt: ChatPromptTemplate, llm: BaseChatModel
) -> Runnable:
    """
    Construye la cadena del analziador de retrieval.
    Args:
        prompt (ChatPromptTemplate): El prompt para el analziador.
        llm (BaseChatModel): El modelo de lenguaje.
    Returns:
        Runnable: La cadena del analziador.
    """

    structured_llm = llm.with_structured_output(RetrievalAnalyzerResponse)

    return prompt | structured_llm
