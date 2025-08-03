"""Dataset service.

This module provides functions to download and manage datasets used
for training the language model.  The implementation here is a stub:
it pretends to download data and records a status variable.  In a
production system this would trigger an asynchronous job in the
ingestion service and update persistent state accordingly.
"""

from typing import Optional

_download_status: Optional[str] = None
_prepare_status: Optional[str] = None


def download_dataset() -> str:
    """Simulate downloading a dataset.

    Returns
    -------
    str
        A message indicating that the download has started.
    """
    global _download_status
    _download_status = "downloading"
    # TODO: trigger asynchronous download via ingestion service
    # For now, pretend download completes instantly
    _download_status = "completed"
    return "Dataset download initiated."


def prepare_dataset() -> str:
    """Prepare the downloaded dataset for training.

    This function reads raw PDFs from ``datasets/papers_raw`` using the
    ingestion parser, performs a simple transformation (extracts text
    and slices it into input/summary pairs) and writes the result to
    ``datasets/training_data/dataset.jsonl``.  The status is recorded
    in a module variable.

    Returns
    -------
    str
        A message indicating that preparation has started.
    """
    global _prepare_status
    _prepare_status = "preparing"
    from pathlib import Path
    import json
    from services.ingestion_service import parser  # local import to avoid circular dep

    raw_dir = Path("datasets/papers_raw")
    output_dir = Path("datasets/training_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "dataset.jsonl"
    with output_file.open("w", encoding="utf-8") as out_f:
        for pdf_path in raw_dir.glob("*.pdf"):
            try:
                text = parser.parse_pdf(pdf_path)  # type: ignore[arg-type]
                # Create a simple training example from the first parts of the text
                input_text = text[:1000]
                summary = text[:200]
                out_f.write(json.dumps({"text": input_text, "summary": summary}) + "\n")
            except Exception as exc:  # pragma: no cover
                print(f"[DatasetService] Failed to process {pdf_path}: {exc}")
    _prepare_status = "completed"
    return "Dataset preparation started."


def get_prepare_status() -> str:
    """Return the current dataset preparation status."""
    return _prepare_status or "not started"


def get_download_status() -> str:
    """Return the current download status."""
    return _download_status or "not started"
