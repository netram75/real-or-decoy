#!/bin/bash
# whole thing in one go: data -> split -> baseline -> reference -> scores
# usage: ./run_all.sh [seed]
set -e
SEED=${1:-7}
PY=${PY:-.venv/bin/python}

$PY generate_data.py --seed "$SEED"
$PY prepare.py --seed "$SEED"

$PY baseline.py public baseline_submission.csv > /dev/null
echo "== baseline"
$PY grade.py --submission baseline_submission.csv --by-split

$PY solution.py public submission.csv > /dev/null
echo "== solution.py"
$PY grade.py --submission submission.csv --by-split
