# Datasets

This directory contains data used and produced by the AI research assistant.  It
is structured to separate raw input files from processed outputs and
metadata.

- **papers_raw/**: PDF or text files retrieved directly from sources (e.g. arXiv,
  conference proceedings).  The ingestion service places newly
  downloaded papers here.  Files may be organised in subdirectories by
  source or date.
- **papers_processed/**: Text extracted from the PDFs and split into chunks for
  embedding.  Keeping processed files allows reprocessing without
  downloading the originals again.
- **metadata/**: Machine‑readable information about papers (titles, authors,
  publication year, etc.).  This may mirror the metadata stored in the
  database or vector store but is useful for offline analysis and
  backups.
- **training_data/**: Any labelled data used for fine‑tuning models.  For
  example, you might place human‑written summaries here for training a
  specialised summariser.
- **user_uploads/**: Documents uploaded by end users.  These are kept
  separate from system‑ingested documents for privacy.

Maintaining a clear data organisation makes it easy to trace the origin
of information and to reproduce experiments.  By keeping raw and
processed data distinct, you can re‑run parts of the pipeline without
polluting the original sources.