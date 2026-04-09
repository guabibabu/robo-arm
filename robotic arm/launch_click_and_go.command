#!/bin/bash

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$ROOT_DIR/launch_common.sh"

cd "$ROOT_DIR" || exit 1
activate_workspace_venv || exit 1
setup_python_runtime || exit 1

export DOBOT_BACKEND=sim
export DOBOT_SIM_DISABLE_VIEWER=1

echo "使用 venv：$DOBOT_WORKSPACE_VENV"
echo "打开离线 Click-and-Go 模拟..."
echo

python3 dobot_rac_workshop-master/scripts/click_and_go_offline.py
