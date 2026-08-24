from pydantic import BaseModel


class RetrievalAnalyzerResponse(BaseModel):
    """
    Respuesta estructurada para el analziador de retrieval.
    Args:
        need_retrieval (bool): Indica si se necesita realizar retrieval.
        confidence (float): Nivel de confianza de la decision tomada (0-1).
        reason (str): La razón que justifica la decision tomada.
    """

    need_retrieval: bool
    confidence: float
    reason: str
