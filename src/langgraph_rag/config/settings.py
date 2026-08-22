import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """
    Configuración de la aplicación (singleton).
    """

    _instance = None

    def __new__(cls) -> "Settings":
        """Crea la instancia si no existe"""
        if cls._instance is None:
            cls._instance = super(Settings, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Inicializa la configuración y valida las variables de entorno"""
        if self._initialized:
            return

        # Validación de variables de entorno
        self.validate()

        # Modelos
        self.LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.EMBEDDING_MODEL: str = os.getenv(
            "EMBEDDING_MODEL", "text-embedding-3-small"
        )

        # Parámetros LLM
        self.DEFAULT_TEMPERATURE: float = 0.5
        self.MAX_RETRIES: int = 3

        # Rutas de persistencia
        self.CHROMA_PATH: str = os.getenv("CHROMA_PATH", "./data/langchain_chroma")
        self.DOCUMENTS_PATH: str = os.getenv("DOCUMENTS_PATH", "./data/documents")

        # RAG
        self.CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 500))
        self.CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 50))
        self.TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", 3))
        self.COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "langchain_rag")

        # Marca la instancia como inicializada
        self._initialized = True

    def validate(self) -> None:
        """Valida las variables de entorno requeridas"""
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not defined")


settings = Settings()
