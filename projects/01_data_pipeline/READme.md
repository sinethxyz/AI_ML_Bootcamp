# 01 — Data Pipeline

Goal: Ingest → clean → validate → save analytics-ready data.

## Must Haves
- CLI entry: `python -m src.pipeline --in data/raw --out data/processed`
- Logging & basic validation (row counts, schema)
- Unit tests in `tests/` (at least 3 meaningful cases)

## Run
1) Create `.venv` & install deps  
2) `pytest -q`  
3) `python -m src.pipeline --help`
