from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader


SUPPORTED_EXTENSIONS = {".txt", ".pdf"}


def load_file(file_path: Path) -> list[Document]:
    """Carga un archivo y lo convierte en una lista de documentos de LangChain

    Args:
        file_path (Path): Ruta del archivo

    Returns:
        list[Document]: Lista de documentos
    """

    if not file_path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")

    extension_file = file_path.suffix.lower()

    if extension_file not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"\n❌ Extensión no soportada: {extension_file}"
            f"\n📚 Usa una de estas extensiones: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    print(f"\n⌛ Cargando: {file_path.name}", end="... ")

    if extension_file == ".pdf":
        loader = PyPDFLoader(str(file_path))
        documents = loader.load()

        for document in documents:
            document.metadata["file_name"] = file_path.name
            document.metadata["file_type"] = "pdf"

    elif extension_file == ".txt":
        loader = TextLoader(str(file_path), encoding="utf-8")
        documents = loader.load()

        for document in documents:
            document.metadata["file_name"] = file_path.name
            document.metadata["file_type"] = "txt"

    print("\n✅ Cargado exitoso")

    return documents


def load_directory(path: str) -> list[Document]:
    """Carga todos los documentos de un directorio y los convierte en una lista de documentos de LangChain

    Args:
        path (str): Ruta del directorio

    Returns:
        list[Document]: Lista de documentos
    """

    directory_path = Path(path)

    if not directory_path.exists():
        directory_path.mkdir(parents=True, exist_ok=True)

        print(f"\n📁 Carpeta creada: {directory_path}")
        print(f"\n📁 Agregar archivos en la carpeta y volver a ejecutar")

        return []

    print(f"\n⌛ Procesando directorio: {directory_path.name}", end="... ")

    file_paths = []

    for extension in SUPPORTED_EXTENSIONS:
        file_paths.extend(directory_path.glob(f"*{extension}"))
        file_paths.extend(directory_path.glob(f"*{extension.upper()}"))

    file_paths = list(set(file_paths))

    if not file_paths:
        print(f"\n❌ No se encontraron archivos en el directorio: {directory_path}")
        print(f"\n📚 Agregar archivos en la carpeta y volver a ejecutar")

        return []

    documents = []
    errors = []

    for file_path in sorted(file_paths):
        try:
            docs = load_file(file_path)
            documents.extend(docs)
        except Exception as e:
            errors.append((file_path.name, str(e)))
            print(f"\n❌ Error cargando {file_path.name}: {e}")

    if errors:
        print(f"\n❌ {len(errors)} archivo(s) con error")
        print(f"\n✅ {len(documents)} documento(s) cargado(s)")
    else:
        print(
            f"\n✅ {len(file_paths)} archivo(s) cargados -> {len(documents)} documento(s)"
        )

    return documents
