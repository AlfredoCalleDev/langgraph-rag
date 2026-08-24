from functools import partial
from langchain_core.runnables import Runnable
from langgraph.graph import START, END, StateGraph
from langchain_core.vectorstores import VectorStore
from langgraph_rag.states.RAGState import RAGState
from langgraph_rag.nodes.retriever_node import retriever_node
from langgraph_rag.nodes.generator_node import generator_node
from langgraph_rag.nodes.retrieval_analyzer_node import retrieval_analyzer_node
from langgraph_rag.chains.analyzer_chain import build_retrieval_analyzer_chain
from langgraph_rag.chains.generator_chain import build_generator_chain
from langgraph_rag.ai.llm import get_llm
from langgraph_rag.prompts.assistant import assistant_prompt
from langgraph_rag.prompts.retrieval_analyzer import retrieval_analyzer_prompt
from langgraph_rag.edges.retrieval_analyzer_edge import retrieval_analyzer_edge


llm = get_llm()


def build_rag_graph(vector_store: VectorStore) -> Runnable:
    """
    Construye el RAG graph.
    Args:
        vector_store (VectorStore): Vector store.
    Returns:
        Runnable: El RAG graph compilado.
    """
    retrieval_analyzer_chain = build_retrieval_analyzer_chain(
        prompt=retrieval_analyzer_prompt, llm=llm
    )
    generator_chain = build_generator_chain(prompt=assistant_prompt, llm=llm)

    rag_graph = StateGraph(RAGState)

    retrieval_analyzer_node_with_chain = partial(
        retrieval_analyzer_node, analyzer_chain=retrieval_analyzer_chain
    )
    retriever_node_with_vector_store = partial(
        retriever_node, vector_store=vector_store
    )
    generator_node_with_chain = partial(generator_node, generator_chain=generator_chain)

    rag_graph.add_node("retrieval_analyzer", retrieval_analyzer_node_with_chain)
    rag_graph.add_node("retriever", retriever_node_with_vector_store)
    rag_graph.add_node("generator", generator_node_with_chain)

    rag_graph.add_edge(START, "retrieval_analyzer")
    rag_graph.add_conditional_edges(
        "retrieval_analyzer",
        retrieval_analyzer_edge,
        {
            "retriever": "retriever",
            "generator": "generator",
        },
    )
    rag_graph.add_edge("retriever", "generator")
    rag_graph.add_edge("generator", END)

    return rag_graph.compile()
