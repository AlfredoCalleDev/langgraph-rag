import hashlib
from langchain_core.documents import Document
from langgraph_rag.databases.vdb import get_or_create_vector_store
from langchain_core.vectorstores import VectorStore
from langgraph_rag.loaders.file_loader import load_directory
from langgraph_rag.splitters.document_splitter import split_documents
from langgraph_rag.config.settings import settings


def calculate_chunk_ids(chunks: list[Document]) -> list[str]:
    """
    Calcula IDs únicos usando un hash del contenido del chunk y sus metadatos.
    Si el contenido no cambia, el ID será exactamente el mismo.

    Args:
        chunks (list[Document]): Lista de chunks

    Returns:
        list[str]: Lista de IDs únicos
    """
    for chunk in chunks:
        # Creamos un string único combinando la fuente y el contenido del chunk
        source = chunk.metadata.get("source", "desconocida")
        content = chunk.page_content
        unique_string = f"{source}:{content}"

        # Generamos un hash seguro y determinista
        chunk_id = hashlib.sha256(unique_string.encode("utf-8")).hexdigest()
        chunk.metadata["id"] = chunk_id

    return [chunk.metadata["id"] for chunk in chunks]


def index_documents(
    documents_path: str = None,
    chroma_path: str = None,
    collection_name: str = None,
) -> tuple[VectorStore, int]:
    """
    Procesa todos los documentos del directorio y los sincroniza con el vector store.
    Evita duplicados y actualiza fragmentos modificados o eliminados.

    Args:
        documents_path (str): Ruta del directorio
        chroma_path (str): Ruta del vector store
        collection_name (str): Nombre de la colección

    Returns:
        tuple: (vector_store, num_documents_added)
    """

    documents = load_directory(path=documents_path or settings.DOCUMENTS_PATH)

    if not documents:
        print(f"\n❌ No hay documentos para indexar")
        return None, 0

    chunks = split_documents(documents)

    if not chunks:
        print(
            f"\n❌ No hay chunks para indexar. Los archivos posiblemente estén vacíos."
        )
        return None, 0

    # Calcular IDs deterministas
    calculate_chunk_ids(chunks)

    vector_store = get_or_create_vector_store(
        collection_name=collection_name or settings.COLLECTION_NAME,
        persist_path=chroma_path or settings.CHROMA_PATH,
    )

    # 1. Obtener los IDs que ya existen en la base de datos
    existing_items = vector_store.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"\n✅ {len(existing_ids)} chunks ya indexados")

    # 2. Identificar los nuevos IDs generados a partir de los archivos actuales
    new_chunks_dict = {chunk.metadata["id"]: chunk for chunk in chunks}
    new_ids_set = set(new_chunks_dict.keys())

    # 3. Eliminar chunks obsoletos (están en la DB pero ya no están en los archivos locales)
    ids_to_delete = existing_ids - new_ids_set
    if ids_to_delete:
        print(
            f"🗑️ Eliminando {len(ids_to_delete)} chunks obsoletos de la base de datos..."
        )
        vector_store.delete(ids=list(ids_to_delete))

    # 4. Agregar chunks nuevos (están en los archivos locales pero no en la DB)
    ids_to_add = new_ids_set - existing_ids
    if ids_to_add:
        print(f"⌛ Indexando {len(ids_to_add)} chunks nuevos...")
        chunks_to_add = [new_chunks_dict[chunk_id] for chunk_id in ids_to_add]
        vector_store.add_documents(chunks_to_add, ids=list(ids_to_add))
        print(f"📚 {len(ids_to_add)} chunks agregados")
    else:
        print("✅ 0 chunks nuevos indexados")

    new_num_chunks = vector_store._collection.count()
    print(f"📚 {new_num_chunks} chunks en total")

    return vector_store, len(ids_to_add)
