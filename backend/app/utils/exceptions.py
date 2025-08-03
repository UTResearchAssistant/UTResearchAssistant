"""Custom exception types for the research assistant.

Defining dedicated exception classes simplifies error handling and
improves readability when raising and catching errors in services and
agents.
"""


class ServiceError(Exception):
    """Base class for service‑level exceptions."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message
