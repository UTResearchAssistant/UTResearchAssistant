"""Training service.

This module simulates a fine‑tuning process for a smaller language
model.  In reality you would integrate with a machine learning
framework (e.g. HuggingFace Transformers) to perform supervised
training on the prepared dataset.  The status is stored in a module
variable for demonstration purposes.
"""

from typing import Optional

_training_status: Optional[str] = None


def start_training() -> str:
    """Simulate starting a training job."""
    global _training_status
    _training_status = "running"
    # TODO: launch training pipeline using prepared dataset
    _training_status = "completed"
    return "Training started."


def get_training_status() -> str:
    """Return the status of the training job."""
    return _training_status or "not started"
