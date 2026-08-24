from typing import TypedDict, Annotated
from langchain_core.documents import Document
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langgraph_rag.schemas.retrieval_analyzer_response import RetrievalAnalyzerResponse


class RAGState(TypedDict):
    """Estado global del RAG"""

    question: str
    messages: Annotated[list[BaseMessage], add_messages]
    retrieved_documents: list[Document]
    retrieval_analyzer_response: RetrievalAnalyzerResponse
    response: str
