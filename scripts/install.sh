#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -c 'import sys,platform; assert (3,10)<=sys.version_info<(3,13), "Нужен Python 3.10–3.12 (задайте PYTHON_BIN)"; assert platform.machine()=="x86_64", "Этот набор зависимостей предназначен для Linux x86_64"'
"$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
printf '\nГотово. Загрузите модель: .venv/bin/eyemonitor-model\nЗапуск: .venv/bin/eyemonitor\n'
