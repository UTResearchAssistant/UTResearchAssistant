"""Wrapper for OpenAI API calls.

This module centralises interaction with OpenAI's API (or other LLM
providers).  By abstracting the call, the rest of the codebase can
remain agnostic of the specific provider used.  In production this
module would handle rate limiting, error retries and model selection.
"""

from typing import Any, List, Optional

from backend.app.core.config import get_settings


def chat_completion(
    messages: List[dict[str, str]],
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.2,
) -> dict[str, Any]:
    """Call the OpenAI ChatCompletion API.

    Parameters
    ----------
    messages : list[dict[str, str]]
        A list of message dicts following the Chat API schema.
    model : str, optional
        The model name to use.  Defaults to ``gpt-3.5-turbo``.
    temperature : float, optional
        Sampling temperature.  Defaults to 0.2.

    Returns
    -------
    dict[str, Any]
        The response from the API.

    Notes
    -----
    The real implementation is omitted; this stub returns a fixed
    response.  When adding integration, read the API key from the
    settings and call the OpenAI Python SDK.
    """
    settings = get_settings()
    api_key: Optional[str] = settings.openai_api_key
    # TODO: import openai and call openai.ChatCompletion.create
    # Placeholder output mimicking OpenAI response structure
    return {
        "id": "chatcmpl-001",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a placeholder response.",
                },
                "finish_reason": "stop",
            }
        ],
    }
