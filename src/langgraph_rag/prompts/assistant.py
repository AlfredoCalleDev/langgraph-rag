from langchain_core.prompts.chat import ChatPromptTemplate


assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Eres un asistente que responde preguntas
                basándote ÚNICAMENTE en el contexto proporcionado.

                Contexto recuperado de los documentos:
                {context}

                Instrucciones:
                - Si la respuesta está en el contexto, respóndela con precisión.
                - Si no está, di: "No encontré esa información en los documentos."
                - Cita la fuente cuando sea posible.
                - No inventes ni supongas información.""",
        ),
        ("human", "{question}"),
    ]
)
