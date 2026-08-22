from langchain_core.documents import Document


def format_docs(docs: list[Document], show_debug: bool = False) -> str:
    """Convierte una lista de documentos de LangChain en una sola cadena de texto.

    Args:
        docs (list[Document]): Lista de documentos
        show_debug (bool): Mostrar información de depuración

    Returns:
        str: Texto formateado
    """

    text = "\n\n---\n\n".join(
        [
            f"[Fuente: {doc.metadata.get('source', 'desconocida')}, "
            f"Página: {doc.metadata.get('page', 'N/A')}]\n {doc.page_content}"
            for doc in docs
        ]
    )

    if show_debug:
        print(f"\n📖 Texto formateado: {text}")

    return text
