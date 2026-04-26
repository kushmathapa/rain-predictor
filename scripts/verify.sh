#!/usr/bin/env bash
set -euo pipefail

echo "1) Unit tests"
./.venv/bin/python -m unittest discover -s tests -p "test_*.py" -v

echo
echo "2) End-to-end (skip EDA)"
./.venv/bin/rainfall-predictor run-all --skip-eda

echo
echo "3) Regenerate LLM context snapshot"
./scripts/make_llm_context.sh

echo
echo "OK"
