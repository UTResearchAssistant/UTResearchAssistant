"""API endpoints for dataset management.

These endpoints allow clients to trigger dataset downloads and check the
status of downloads.  In a full implementation the download would be
performed asynchronously by the ingestion service.  Here we simulate
the behaviour with a simple function call.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...services import dataset_service


router = APIRouter()


class DatasetDownloadResponse(BaseModel):
    message: str


@router.post("/dataset/download", response_model=DatasetDownloadResponse)
async def download_dataset() -> DatasetDownloadResponse:
    """Trigger a dataset download and return a status message."""
    try:
        msg = dataset_service.download_dataset()
        return DatasetDownloadResponse(message=msg)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc))


class DatasetStatusResponse(BaseModel):
    status: str
    prepare_status: str


@router.get("/dataset/status", response_model=DatasetStatusResponse)
async def dataset_status() -> DatasetStatusResponse:
    """Return the current status of the dataset download."""
    status = dataset_service.get_download_status()
    prepare_status = dataset_service.get_prepare_status()
    return DatasetStatusResponse(status=status, prepare_status=prepare_status)


class DatasetPrepareResponse(BaseModel):
    message: str


@router.post("/dataset/prepare", response_model=DatasetPrepareResponse)
async def prepare_dataset() -> DatasetPrepareResponse:
    """Trigger preparation of the downloaded dataset."""
    try:
        msg = dataset_service.prepare_dataset()
        return DatasetPrepareResponse(message=msg)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc))
