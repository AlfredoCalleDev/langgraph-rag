import warnings
from langgraph_rag.schemas.retrieval_analyzer_response import RetrievalAnalyzerResponse
from langchain_core.messages import HumanMessage, AIMessageChunk
from langgraph_rag.states.RAGState import RAGState
from langgraph_rag.graphs.rag_graph import build_rag_graph
from langgraph_rag.indexers.document_indexer import index_documents
from langgraph_rag.config.settings import settings

# Ocultamos el warning molesto de Pydantic/Langchain
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")


def start_app():
    vector_store, num_chunks = index_documents(
        documents_path=settings.DOCUMENTS_PATH,
        chroma_path=settings.CHROMA_PATH,
        collection_name=settings.COLLECTION_NAME,
    )

    if vector_store is None:
        print("\n Pasos para empezar")
        print(f" 1. Crea la carpera: {settings.DOCUMENTS_PATH}")
        print(" 2. Agrega archivos .txt o .pdf")
        print(" 3. Vuelve a ejecutar este script")
        return

    rag_graph = build_rag_graph(vector_store)

    while True:
        user_question = input("\n👤 Pregunta: ").strip()

        initial_state = RAGState(
            question=user_question,
            messages=[HumanMessage(content=user_question)],
            retrieved_documents=[],
            retrieval_analyzer_response=RetrievalAnalyzerResponse(
                need_retrieval=False, confidence=0.0, reason=""
            ),
            response="",
        )

        print("\n🤖 IA: ", end="")
        # Iteramos sobre el stream del grafo buscando mensajes en vivo
        for msg, metadata in rag_graph.stream(initial_state, stream_mode="messages"):
            # Filtramos: Solo queremos mensajes del generador Y que sean Chunks (pedacitos en vivo)
            if metadata.get("langgraph_node") == "generator" and isinstance(
                msg, AIMessageChunk
            ):
                if msg.content and isinstance(msg.content, str):
                    print(msg.content, end="", flush=True)

        print("\n")


if __name__ == "__main__":
    start_app()
