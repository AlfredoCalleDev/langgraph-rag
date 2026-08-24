from langchain_core.prompts.chat import ChatPromptTemplate


assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Eres un asistente que responde preguntas

                Contexto recuperado de los documentos:
                {context}

                Instrucciones:
                - Si la respuesta está en el contexto, respóndela con precisión.
                - Si no está, di: "No encontré esa información en los documentos."
                - Si el contexto está vacío o la pregunta es de cultura general, responde usando tu propio conocimiento pero aclara que la información no proviene de los documentos. Si hay contexto, básate en él.
                - Cita la fuente cuando sea posible.
                - No inventes ni supongas información.
                """,
        ),
        ("human", "{question}"),
    ]
)
