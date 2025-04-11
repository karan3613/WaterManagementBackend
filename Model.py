from typing import Optional

from pydantic import BaseModel

class ChatRequest(BaseModel):
    question : str
    language : str

# Request schema using Pydantic
class PredictionRequest(BaseModel):
    PH: Optional[float]
    EC: Optional[float]
    ORP: Optional[float]
    DO: Optional[float]
    TDS: Optional[float]
    TSS: Optional[float]
    TS: Optional[float]
    TOTAL_N: Optional[float]
    NH4_N: Optional[float]
    TOTAL_P: Optional[float]
    PO4_P: Optional[float]
    COD: Optional[float]
    BOD: Optional[float]