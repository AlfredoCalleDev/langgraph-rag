from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph_rag.config.settings import settings


def split_documents(
    documents: list[Document],
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> list[Document]:
    """
    Divide los documentos en chunks

    Args:
        documents (list[Document]): Lista de documentos
        chunk_size (int): Tamaño de los chunks
        chunk_overlap (int): Superposición de los chunks

    Returns:
        list[Document]: Lista de documentos divididos en chunks
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.CHUNK_SIZE,
        chunk_overlap=chunk_overlap or settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)

    print(f"📊 Se dividió en {len(chunks)} chunks")

    return chunks
