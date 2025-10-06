# FAQ / Troubleshooting

**Large files rejected on push**  
→ Never commit `.venv/`, data, or models. Use `.gitignore`.

**Black/Ruff not running**  
→ Ensure extensions installed; reload VS Code.

**Interpreter wrong**  
→ Command Palette → Python: Select Interpreter → pick `.venv`.

**Jupyter kernel missing**  
→ `pip install ipykernel` then restart notebook.

**Git auth prompts**  
→ Use a GitHub Personal Access Token with HTTPS or set up SSH keys.
