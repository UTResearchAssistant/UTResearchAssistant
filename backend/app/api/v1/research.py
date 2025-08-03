"""API for deep research queries.

This endpoint performs a multi‑step research process: it uses the
browsing agent to search for relevant information, then summarises
the results using the summariser agent.  The response returns a
consolidated answer and references.  This feature demonstrates how
agents can be orchestrated to provide richer answers than simple
search or summarisation alone.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...services import research_service


router = APIRouter()


class ResearchRequest(BaseModel):
    query: str


class ResearchResponse(BaseModel):
    answer: str
    sources: list[str]


@router.post("/research", response_model=ResearchResponse)
async def perform_research(request: ResearchRequest) -> ResearchResponse:
    """Execute a deep research query using multiple agents."""
    try:
        result = research_service.deep_research(request.query)
        return ResearchResponse(**result)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc))
