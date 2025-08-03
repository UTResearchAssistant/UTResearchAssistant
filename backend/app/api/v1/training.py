"""API endpoints for model training.

These endpoints allow clients to start a training job for a smaller
language model and check its status.  Training is simulated in this
scaffold; a real implementation would involve orchestrating a
fine‑tuning job using HuggingFace or a similar library.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...services import training_service


router = APIRouter()


class TrainingResponse(BaseModel):
    message: str


@router.post("/train/start", response_model=TrainingResponse)
async def start_training() -> TrainingResponse:
    """Begin a model training job."""
    try:
        msg = training_service.start_training()
        return TrainingResponse(message=msg)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc))


class TrainingStatusResponse(BaseModel):
    status: str


@router.get("/train/status", response_model=TrainingStatusResponse)
async def training_status() -> TrainingStatusResponse:
    """Get the status of the ongoing or last training job."""
    return TrainingStatusResponse(status=training_service.get_training_status())
