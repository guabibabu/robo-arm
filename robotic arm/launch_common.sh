#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSHOP_DIR="$ROOT_DIR/dobot_rac_workshop-master"

activate_workspace_venv() {
  local candidate
  for candidate in "$ROOT_DIR/.venv" "$ROOT_DIR/venv" "$ROOT_DIR/env"; do
    if [ -f "$candidate/bin/activate" ]; then
      # shellcheck disable=SC1090
      source "$candidate/bin/activate"
      export DOBOT_WORKSPACE_VENV="$candidate"
      return 0
    fi
  done

  echo "找不到可用的 virtualenv。"
  echo "请先在工作区建立 .venv / venv / env。"
  return 1
}

setup_python_runtime() {
  export MPLCONFIGDIR="$ROOT_DIR/.cache/matplotlib"
  mkdir -p "$MPLCONFIGDIR"

  if ! python3 -c "import matplotlib" >/dev/null 2>&1; then
    echo "目前这个 venv 里还没有 matplotlib。"
    echo "请先运行："
    echo "  source \"$DOBOT_WORKSPACE_VENV/bin/activate\""
    echo "  pip install -r \"$ROOT_DIR/requirements.txt\""
    return 1
  fi
}

run_offline_sim_demo() {
  local demo="$1"
  local script_path

  case "$demo" in
    arm)
      script_path="$WORKSHOP_DIR/scripts/arm_move.py"
      ;;
    pick)
      script_path="$WORKSHOP_DIR/scripts/manual_customized_task.py"
      ;;
    *)
      echo "未知的 demo：$demo"
      return 1
      ;;
  esac

  cd "$ROOT_DIR"
  activate_workspace_venv
  setup_python_runtime

  export DOBOT_BACKEND=sim

  echo "使用 venv：$DOBOT_WORKSPACE_VENV"
  echo "启动离线机械臂模拟：$demo"
  echo

  python3 "$script_path"
}
