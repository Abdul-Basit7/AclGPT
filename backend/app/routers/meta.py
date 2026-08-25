from typing import List

from fastapi import APIRouter

from ..config import settings
from ..schemas import HealthOut, ModelInfo
from ..services import llm as llm_service

router = APIRouter(prefix="/api", tags=["meta"])


@router.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    return HealthOut(
        status="ok",
        google_key_configured=bool(settings.google_api_key),
        groq_key_configured=bool(settings.groq_api_key),
        models=llm_service.list_models(),
    )


@router.get("/models", response_model=List[ModelInfo])
def models() -> List[ModelInfo]:
    return llm_service.list_models()
