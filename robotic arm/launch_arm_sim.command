#!/bin/bash

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$ROOT_DIR/launch_common.sh"

cd "$ROOT_DIR" || exit 1
activate_workspace_venv || exit 1
setup_python_runtime || exit 1

export DOBOT_BACKEND=sim

echo "使用 venv：$DOBOT_WORKSPACE_VENV"
echo "打开离线机械臂快捷编辑器..."
echo

exec python3 "$ROOT_DIR/dobot_rac_workshop-master/scripts/arm_sim_control_panel.py"
