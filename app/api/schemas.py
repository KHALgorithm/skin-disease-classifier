from typing import  List

from pydantic import BaseModel, Field


class Base64Image(BaseModel):
    image: str = Field(..., description="Base64 encoded image")


class ClassificationResponse(BaseModel):
    label: str
    confidence: float


class BatchClassificationResponse(BaseModel):
    results: List[ClassificationResponse]
