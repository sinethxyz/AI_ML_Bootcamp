#!/usr/bin/env bash
set -e

mkdir -p week_01_python/day_01_basics \
         week_01_python/day_02_control_flow \
         week_01_python/day_03_data_structures \
         week_01_python/day_04_comprehensions \
         week_01_python/day_05_error_handling \
         week_01_python/weekend_project_etl \
         week_02_numpy_pandas \
         week_03_math \
         week_04_eda \
         projects/01_data_pipeline \
         projects/02_etl_system \
         notes/daily_logs \
         notes/patterns_discovered \
         notes/gotchas \
         resources/datasets \
         resources/papers \
         resources/references \
         tests

[ ! -f README.md ] && cat > README.md << 'R2'
# AI_ML_Bootcamp
(See repo root README in your current project—customize me!)
R2

[ ! -f progress_tracker.md ] && cat > progress_tracker.md << 'R3'
# Progress Tracker
| Week | Focus | Status | Notes |
|------|-------|--------|-------|
| 01 | Python Foundations | ☐ | |
| 02 | NumPy & Pandas | ☐ | |
| 03 | Math & Probability | ☐ | |
| 04 | EDA & Visualization | ☐ | |
R3

[ ! -f requirements.txt ] && cat > requirements.txt << 'R4'
black
pylint
jupyter
numpy
pandas
matplotlib
scikit-learn
R4

echo "Scaffold complete."
