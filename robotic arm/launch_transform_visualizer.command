#!/bin/bash

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$ROOT_DIR/launch_common.sh"

cd "$ROOT_DIR" || exit 1
activate_workspace_venv || exit 1
setup_python_runtime || exit 1

if ! python3 -c "import matplotlib, numpy, scipy" >/dev/null 2>&1; then
  echo "目前這個 venv 缺少需要的 Python 套件。"
  echo "請先執行："
  echo "  source \"$DOBOT_WORKSPACE_VENV/bin/activate\""
  echo "  pip install matplotlib numpy scipy"
  read -r -p "按 Enter 關閉..."
  exit 1
fi

python3 dobot_rac_workshop-master/calibration/visualize_transform_chain.py
