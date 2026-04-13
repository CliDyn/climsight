"""Lightweight REST routes for the deployable ClimAssist MVP."""

from __future__ import annotations

from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException

from climassist_service import AdvisoryRequest, generate_advisory, get_runtime_status

router = APIRouter()


class ClimAssistAnalyzeRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    crop: str = Field(default="millet", min_length=1, max_length=80)
    stage: str = Field(default="planting", min_length=1, max_length=80)
    question: str = Field(default="", max_length=1200)
    language: str = Field(default="english", min_length=2, max_length=40)
    model_name: str | None = Field(default=None, max_length=120)
    api_key: str | None = Field(default=None, max_length=300)


@router.get("/climassist/status")
async def climassist_status():
    return get_runtime_status()


@router.post("/climassist/analyze")
async def climassist_analyze(payload: ClimAssistAnalyzeRequest):
    try:
        request = AdvisoryRequest(
            lat=payload.lat,
            lon=payload.lon,
            crop=payload.crop.strip(),
            stage=payload.stage.strip(),
            question=payload.question.strip(),
            language=payload.language.strip().lower(),
            model_name=payload.model_name,
            api_key=payload.api_key,
        )
        return generate_advisory(request)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Advisory generation failed: {exc}") from exc
