"""Health check endpoint.

This simple router exposes a `/health` endpoint that clients or load
balancers can ping to verify the service is running.  It does not
depend on any external resources.
"""

from fastapi import APIRouter


router = APIRouter()


@router.get("/health", tags=["health"])
async def health() -> dict[str, str]:
    """Return a basic health status.

    Returns
    -------
    dict[str, str]
        A simple JSON object indicating the service is alive.
    """
    return {"status": "ok"}
