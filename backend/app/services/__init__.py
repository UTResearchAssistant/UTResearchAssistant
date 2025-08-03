"""Business logic layer.

Services encapsulate the core functionality of the application and
coordinate interactions between agents, models and the database.  By
placing logic here, the API layer remains thin and easy to test.
"""

from .summarization_service import summarise_text  # noqa: F401
from .search_service import search_documents  # noqa: F401
from .dataset_service import download_dataset, get_download_status  # noqa: F401
from .training_service import start_training, get_training_status  # noqa: F401
from .research_service import deep_research  # noqa: F401

__all__ = ["summarise_text", "search_documents"]

__all__.extend(
    [
        "download_dataset",
        "get_download_status",
        "prepare_dataset",
        "get_prepare_status",
        "start_training",
        "get_training_status",
        "deep_research",
    ]
)
