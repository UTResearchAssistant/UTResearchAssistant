"""Base class for agents.

Agents encapsulate complex multi‑step reasoning or interactions with
external tools.  Each agent implements a ``run`` method that accepts
a query and returns a result.  Subclasses should override ``run``
and may expose additional helper methods.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    @abstractmethod
    def run(self, query: str, **kwargs: Any) -> Any:
        """Execute the agent's logic.

        Parameters
        ----------
        query : str
            The user query or instruction.
        **kwargs : Any
            Additional keyword arguments for the agent.

        Returns
        -------
        Any
            The result of the agent's operation.
        """
        raise NotImplementedError
