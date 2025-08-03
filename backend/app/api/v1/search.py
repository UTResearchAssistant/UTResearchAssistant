"""API router for search operations.

The search endpoints allow clients to query the knowledge base for
relevant papers or snippets.  In this scaffold, the search logic is
delegated to ``search_service`` which currently returns a dummy list.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

# Import the service from the parent package's services module.  We go up two
# levels (from api/v1 to app) to find the services package.
from ...services import search_service


router = APIRouter()


class SearchResponse(BaseModel):
    """Response model for search results.

    Attributes
    ----------
    results : list[str]
        A list of identifiers or titles of matching documents.
    """

    results: list[str]


@router.get("/search", response_model=SearchResponse)
async def search(q: str = Query(..., description="Search query")) -> SearchResponse:
    """Search for documents matching a query string.

    Parameters
    ----------
    q : str
        The search term provided by the client.

    Returns
    -------
    SearchResponse
        The list of matching document identifiers.
    """
    try:
        results = search_service.search_documents(q)
        return SearchResponse(results=results)
    except Exception as exc:  # pragma: no cover - placeholder
        raise HTTPException(status_code=500, detail=str(exc))
