# 🧠 LangGraph RAG

Un sistema RAG (Retrieval-Augmented Generation) avanzado de línea de comandos, construido con **LangGraph**, **LangChain**, OpenAI y Chroma DB. Este proyecto se destaca por no ser un RAG lineal estándar; implementa un flujo de trabajo inteligente (grafo) que **evalúa dinámicamente** si necesita consultar tus documentos o si puede responder directamente usando conocimiento general, optimizando así los recursos y mejorando los tiempos de respuesta.

## ✨ Características Principales

- **Enrutamiento Inteligente (LangGraph)**: Utiliza un nodo analista (`retrieval_analyzer_node`) que decide condicionalmente si invocar la base de datos vectorial o pasar directo a la generación.
- **Sincronización Inteligente de Chunks**: Utiliza hashes criptográficos (SHA-256) para identificar de forma única cada fragmento de texto. Si editas un documento, el sistema detecta los cambios, elimina los fragmentos obsoletos y agrega los nuevos, evitando la duplicación y ahorrando costos de llamadas a la API de Embeddings.
- **Soporte Multiformato**: Ingesta automática de archivos `.txt` y `.pdf` utilizando cargadores especializados.
- **Streaming en Tiempo Real**: Efecto de máquina de escribir real. Intercepta los _chunks_ directamente desde el LLM utilizando `stream_mode="messages"`.
- **Salidas Estructuradas (Pydantic)**: Obliga al LLM a devolver respuestas predecibles en formato JSON para la toma de decisiones usando `.with_structured_output()`.
- **Arquitectura Modular**: El código está fuertemente dividido por responsabilidades (Nodos, Aristas, Loaders, Indexers, Splitters, Chains y Prompts).

## 🛠️ Requisitos Previos

- **Python**: 3.12 o superior.
- **API Key de OpenAI**: Necesaria para el modelo de lenguaje y la generación de embeddings.
- Gestor de paquetes [uv](https://github.com/astral-sh/uv) (recomendado) o `pip`.

## 🚀 Instalación y Configuración

**1. Clonar el repositorio:**

```bash
git clone https://github.com/tu-usuario/langgraph-rag.git
cd langgraph-rag
```

**2. Instalar dependencias:**
Si utilizas el gestor rápido `uv` (Recomendado):

```bash
uv sync
```

Alternativamente, usando `pip` nativo:

```bash
python -m venv .venv # Activar el entorno virtual (En Windows: .venv\Scripts\activate)
pip install -e .
```

**3. Configurar Variables de Entorno:**
El sistema depende de variables de entorno para funcionar. Copia el archivo de ejemplo y agrega tu clave de OpenAI:

```bash
cp .env.example .env
```

Abre el archivo `.env` y asegúrate de configurar tu `OPENAI_API_KEY`:

```env
OPENAI_API_KEY="sk-tu-api-key-aqui"
LLM_MODEL="gpt-4o-mini"
EMBEDDING_MODEL="text-embedding-3-small"
```

## 📚 Uso

**1. Agregar documentos:**
Coloca todos los archivos `.txt` o `.pdf` que desees consultar dentro de la carpeta `data/documents/` (si la carpeta no existe, el programa la creará en su primera ejecución).

**2. Ejecutar la aplicación:**
A través del script definido en tu entorno:

```bash
uv run langgraph-rag
```

O ejecutando el archivo principal con Python:

```bash
python src/langgraph_rag/main.py
```

**3. Interactuar:**
Una vez procesados e indexados los documentos (el sistema te indicará cuántos chunks se actualizaron, agregaron o eliminaron), se abrirá la terminal:

```text
👤 Pregunta: ¿Qué información tienes sobre [X]?
🤖 IA: Según los documentos, la información es...
```

Para salir, simplemente presiona `Ctrl + C`.

## 🏗️ Estructura de carpetas del proyecto

```text
langgraph-rag/
├── data/
│   ├── documents/             # 📥 Coloca aquí tus archivos .txt y .pdf
│   └── langchain_chroma/      # 🗄️ Base de datos vectorial de Chroma (Autogenerada)
├── src/langgraph_rag/
│   ├── ai/                    # Proveedores de LLM y Embeddings
│   ├── chains/                # Cadenas LCEL (Analyzer y Generator)
│   ├── config/                # Gestión de configuraciones y .env
│   ├── databases/             # Conexión e inicialización de ChromaDB
│   ├── edges/                 # Lógica de las aristas condicionales del grafo
│   ├── graphs/                # Ensamblaje del StateGraph de LangGraph
│   ├── indexers/              # Lógica de Hashing y Sincronización de Base de Datos
│   ├── loaders/               # Ingesta de archivos del disco
│   ├── nodes/                 # Nodos funcionales (Analyzer, Retriever, Generator)
│   ├── prompts/               # System Prompts para el asistente
│   ├── schemas/               # Clases Pydantic para salidas estructuradas
│   ├── splitters/             # Chunking de documentos
│   ├── states/                # Definición del RAGState (TypedDict)
│   ├── utils/                 # Formateadores auxiliares
│   └── main.py                # Punto de entrada
├── pyproject.toml             # Configuración de dependencias y scripts
└── .env                       # Variables de entorno
```

## ⚙️ Configuración Avanzada

Puedes modificar los parámetros de RAG dentro de `src/langgraph_rag/config/settings.py` o sobrescribirlos usando el archivo `.env`:

- `CHUNK_SIZE` (default: 500)
- `CHUNK_OVERLAP` (default: 50)
- `TOP_K_RESULTS` (default: 3)

## 📝 Notas de Desarrollo

- **Warnings de Pydantic**: Existe un warning conocido de Pydantic V2 (`PydanticSerializationUnexpectedValue`) que aparece al intentar streamear salidas estructuradas JSON en vivo. Este warning ha sido suprimido de forma intencionada en `main.py` mediante el módulo `warnings`, ya que es un bug interno en la serialización de `langchain-core` que no afecta el comportamiento de esta aplicación.
