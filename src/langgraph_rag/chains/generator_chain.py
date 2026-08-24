from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser


def build_generator_chain(prompt: ChatPromptTemplate, llm: BaseChatModel) -> Runnable:
    """
    Construye la cadena del generador de respuestas.
    Args:
        prompt (ChatPromptTemplate): El prompt para el generador.
        llm (BaseChatModel): El modelo de lenguaje.
    Returns:
        Runnable: La cadena del generador.
    """
    return prompt | llm | StrOutputParser()
