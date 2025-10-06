
#!/usr/bin/env bash
set -e
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pytest ipykernel
python -m ipykernel install --user --name ai-ml-bootcamp >/dev/null 2>&1 || true
echo "Env ready. VS Code → select '.venv' interpreter."
