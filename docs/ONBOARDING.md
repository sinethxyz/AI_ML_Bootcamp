# Onboarding & Environment Setup

## Prerequisites
- Python 3.11+, Git, VS Code (+ Python, Jupyter, Black)

## Setup
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

Open VS Code → Python: Select Interpreter → choose `.venv`.
Auto-save/format/lint is configured in `.vscode/settings.json`.

## Common Commands
- Run a script: `python path/to/file.py`
- Run tests: `pytest` (after `pip install pytest`)
- Freeze deps: `pip freeze > requirements.txt`
