#!/usr/bin/env bash
# All of this file is ai-assisted
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
BOOTSTRAP_STATE_DIR="${ROOT_DIR}/.bootstrap"
BOOTSTRAP_PID_DIR="${BOOTSTRAP_STATE_DIR}/pids"
BOOTSTRAP_LOG_DIR="${BOOTSTRAP_STATE_DIR}/logs"

SERVICE_DIR="${ROOT_DIR}/service"
FRONTEND_DIR="${ROOT_DIR}/frontend"
INFRA_DIR="${ROOT_DIR}/infra/mlflow"

BACKEND_MODULE="backend.main:app"
BACKEND_HOST="0.0.0.0"
BACKEND_PORT="8000"
BACKEND_URL="http://${BACKEND_HOST}:${BACKEND_PORT}"

FRONTEND_HOST="0.0.0.0"
FRONTEND_PORT="5173"
FRONTEND_URL="http://${FRONTEND_HOST}:${FRONTEND_PORT}"

JUPYTER_HOST="0.0.0.0"
JUPYTER_PORT="8888"
JUPYTER_URL="http://${JUPYTER_HOST}:${JUPYTER_PORT}"

PROCESS_POLL_INTERVAL="${PROCESS_POLL_INTERVAL:-5}"

# Choose python for uv venv (override: PYTHON_SPEC=3.12 ./bootstrap.sh)
PYTHON_SPEC="${PYTHON_SPEC:-3.13}"

# ---------- helpers ----------
has_dir() { [[ -d "$1" ]]; }
has_cmd() { command -v "$1" >/dev/null 2>&1; }

pause() { read -r -p "Press Enter to continue..." _; }

# ---------- uv / venv ----------
ensure_uv() {
  if has_cmd uv; then return 0; fi

  echo "▶ uv is not installed. Installing to ~/.local/bin ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh

  # make available in current shell
  export PATH="$HOME/.local/bin:$PATH"

  if ! has_cmd uv; then
    echo "✖ uv installed but not found in PATH."
    echo "  Add to your shell rc: export PATH=\"\$HOME/.local/bin:\$PATH\""
    return 1
  fi
}

uv_in_venv() {
  if [[ ! -d "${VENV_DIR}" ]]; then
    echo "▶ Creating uv venv at ${VENV_DIR} (python: ${PYTHON_SPEC})"
    uv_cmd venv "${VENV_DIR}" --python "${PYTHON_SPEC}"
  fi
}

uv_cmd() {
  ensure_uv
  uv "$@"
}

uv_pip() {
  # Run uv pip against our venv python (IMPORTANT: uv pip has no --python flag)
  uv_in_venv
  UV_PYTHON="${VENV_DIR}/bin/python" uv_cmd pip "$@"
}

uv_run() {
  # Run command inside our venv python
  uv_in_venv
  uv_cmd run --python "${VENV_DIR}/bin/python" "$@"
}


# Run long-lived command so menu doesn't die.
# We keep your desired behavior: command runs, you Ctrl+C or it exits -> back to menu.
run_and_wait() {
  set +e
  "$@"
  local status=$?
  set -e
  [[ ${status} -eq 130 ]] && return 0
  return "${status}"
}

run_and_wait_in_dir() {
  local dir="$1"
  shift
  set +e
  ( cd "${dir}" && "$@" )
  local status=$?
  set -e
  [[ ${status} -eq 130 ]] && return 0
  return "${status}"
}

# ---------- installers ----------
install_python_base() {
  uv_in_venv
  echo "▶ Installing project (editable): ${ROOT_DIR}"
  uv_pip install -e "${ROOT_DIR}"
  echo "✔ project installed"
}

install_python_dev() {
  uv_in_venv
  echo "▶ Installing project with dev extras: ${ROOT_DIR}[dev]"
  uv_pip install -e "${ROOT_DIR}[dev]"
  echo "✔ project[dev] installed"
}

install_python_notebooks() {
  uv_in_venv
  echo "▶ Installing project with notebooks extras: ${ROOT_DIR}[notebooks]"
  uv_pip install -e "${ROOT_DIR}[notebooks]"
  echo "✔ project[notebooks] installed"
}

install_python_all() {
  uv_in_venv
  echo "▶ Installing project with all extras: ${ROOT_DIR}[dev,notebooks,viz,cv]"
  uv_pip install -e "${ROOT_DIR}[dev,notebooks,viz,cv]"
  echo "✔ project[dev,notebooks,viz,cv] installed"
}

# ---------- runners ----------
run_backend() {
  install_python_base >/dev/null

  echo "▶ Running backend: uvicorn ${BACKEND_MODULE} --reload"
  run_and_wait_in_dir "${SERVICE_DIR}" \
    uv_run python -m uvicorn "${BACKEND_MODULE}" --reload --host "${BACKEND_HOST}" --port "${BACKEND_PORT}"
}

alembic_upgrade() {
  install_python_base >/dev/null
  echo "▶ Alembic upgrade head"
  ( cd "${SERVICE_DIR}" && uv_run python -m alembic upgrade head )
  echo "✔ Alembic upgraded"
}

alembic_revision() {
  install_python_base >/dev/null
  read -r -p "Migration message: " MSG
  echo "▶ Alembic revision --autogenerate -m \"${MSG}\""
  ( cd "${SERVICE_DIR}" && uv_run python -m alembic revision --autogenerate -m "${MSG}" )
  echo "✔ Alembic revision created"
}

run_jupyter() {
  install_python_notebooks >/dev/null

  echo "▶ Starting JupyterLab"
  run_and_wait_in_dir "${ROOT_DIR}" \
    uv_run jupyter-lab --no-browser --ip="${JUPYTER_HOST}" --port="${JUPYTER_PORT}"
}

publish_models_to_wandb() {
  install_python_base >/dev/null

  if ! has_dir "${ROOT_DIR}/models"; then
    echo "✖ models/ not found at: ${ROOT_DIR}/models"
    return 1
  fi

  echo "▶ Publishing models from models/ to W&B"
  echo "  Uses WANDB_PROJECT/WANDB_ENTITY if set; otherwise asks interactively."
  run_and_wait_in_dir "${ROOT_DIR}" \
    uv_run python "${ROOT_DIR}/scripts/publish_models_to_wandb.py" --models-dir "${ROOT_DIR}/models"
}

download_best_model() {
  install_python_base >/dev/null

  local dest="${MODEL_DOWNLOAD_DIR:-${ROOT_DIR}/models/mlflow}"
  echo "▶ Downloading best MURA model from the MLflow registry"
  echo "  Model + tracking URI are taken from .env (MLFLOW_MODEL_NAME/VERSION/ALIAS)"
  echo "  Destination: ${dest}"
  mkdir -p "${dest}"

  run_and_wait_in_dir "${ROOT_DIR}" \
    uv_run python -c '
import os
import sys

from dotenv import load_dotenv

load_dotenv()

import mlflow
from mlflow.artifacts import download_artifacts

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
model_name = os.getenv("MLFLOW_MODEL_NAME", "mura_dinov2_transformer")
model_version = os.getenv("MLFLOW_MODEL_VERSION", "").strip()
model_alias = os.getenv("MLFLOW_MODEL_ALIAS", "prd")
model_uri = os.getenv("MLFLOW_MODEL_URI", "").strip()

if not model_uri:
    if model_version:
        model_uri = f"models:/{model_name}/{model_version}"
    else:
        model_uri = f"models:/{model_name}@{model_alias}"

dest = sys.argv[1]

mlflow.set_tracking_uri(tracking_uri)
print(f"Resolving {model_uri} from {tracking_uri}")
local_path = download_artifacts(artifact_uri=model_uri, dst_path=dest)
print(f"Model artifacts downloaded to: {local_path}")
' "${dest}"
}

upload_model_to_mlflow() {
  install_python_base >/dev/null

  local src="${MODEL_UPLOAD_PATH:-}"
  if [[ -z "${src}" ]]; then
    read -r -p "Path to model file (.pt/.pth) or MLflow model dir: " src
  fi
  if [[ -z "${src}" ]]; then
    echo "✖ No path provided"
    return 1
  fi
  if [[ ! -e "${src}" ]]; then
    echo "✖ Path not found: ${src}"
    return 1
  fi

  echo "▶ Registering model into MLflow from: ${src}"
  echo "  Registry name + alias are taken from .env (MLFLOW_MODEL_NAME/MLFLOW_MODEL_ALIAS)"

  # The MLflow server writes run artifacts straight to MinIO (s3://), so the
  # client needs S3 credentials + endpoint. Pull them from the infra .env
  # unless the caller already exported their own.
  local s3_port minio_user minio_pass
  s3_port="$(infra_env_value S3_API_PORT 9900)"
  minio_user="$(infra_env_value MINIO_ROOT_USER admin)"
  minio_pass="$(infra_env_value MINIO_ROOT_PASSWORD password)"

  (
    export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-${minio_user}}"
    export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-${minio_pass}}"
    export MLFLOW_S3_ENDPOINT_URL="${MLFLOW_S3_ENDPOINT_URL:-http://localhost:${s3_port}}"
    echo "  S3 endpoint: ${MLFLOW_S3_ENDPOINT_URL}"

  run_and_wait_in_dir "${ROOT_DIR}" \
    uv_run python -c '
import os
import sys

from dotenv import load_dotenv

load_dotenv()

import mlflow
from mlflow.tracking import MlflowClient

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
model_name = os.getenv("MLFLOW_MODEL_NAME", "mura_dinov2_transformer")
model_alias = os.getenv("MLFLOW_MODEL_ALIAS", "prd")

src = sys.argv[1]
src_label = os.path.basename(src.rstrip("/"))
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(model_name + "_uploads")

pip_requirements = [
    "torch",
    "torchvision",
    "transformers",
    "mlflow>=2.15.1,<3",
    "numpy",
    "pandas",
    "pillow",
    "scikit-learn",
]


def is_mlflow_model_dir(path):
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "MLmodel"))


version = ""
if is_mlflow_model_dir(src):
    # Already an MLflow-packaged model: copy the artifacts as-is and register a
    # new version. We deliberately do NOT re-pickle the model object here -
    # the notebook classifier class lives in __main__ and cloudpickle tries to
    # serialize it by value, which crashes on Python 3.13 (dis IndexError).
    print("Detected MLflow model directory; registering without re-pickling")
    with mlflow.start_run(run_name="upload_" + src_label) as run:
        mlflow.set_tag("uploaded_from", src)
        mlflow.log_artifacts(src, artifact_path="model")
        run_id = run.info.run_id
    result = mlflow.register_model(model_uri="runs:/" + run_id + "/model", name=model_name)
    version = str(result.version)
else:
    # Raw torch object saved via torch.save(model, path). This path re-pickles
    # through mlflow.pytorch and may fail for notebook-defined classes.
    import torch
    import torch.nn as nn

    # Make the notebook-pickled classifier importable as __main__.MuraDinoClassifier
    try:
        import ml.dino_predictor as dino_predictor
        sys.modules["__main__"].MuraDinoClassifier = dino_predictor.MuraDinoClassifier
    except Exception as exc:
        print("warning: could not register pickle alias:", exc)

    print("Loading torch-saved model object from file")
    try:
        model = torch.load(src, map_location="cpu", weights_only=False)
    except TypeError:
        model = torch.load(src, map_location="cpu")

    if not isinstance(model, nn.Module):
        raise SystemExit(
            "Provided file is not a full torch model (got %s). "
            "Save the whole model via torch.save(model, path), or pass an MLflow model dir."
            % type(model).__name__
        )

    model.eval()
    with mlflow.start_run(run_name="upload_" + src_label):
        mlflow.set_tag("uploaded_from", src)
        info = mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name=model_name,
            pip_requirements=pip_requirements,
        )
    version = str(getattr(info, "registered_model_version", "") or "")

client = MlflowClient()
if version:
    client.set_model_version_tag(model_name, version, "env", "PRD")
    client.set_model_version_tag(model_name, version, "uploaded_from", src_label)
    client.set_registered_model_alias(model_name, model_alias, version)
    print("Registered model:", model_name, "version:", version, "alias:", model_alias)
    print("Note: backend resolves MLFLOW_MODEL_VERSION first; set it to", version,
          "or clear it to use the alias", model_alias)
else:
    print("Model logged but no registered version was returned")
' "${src}"
  )
}

run_frontend() {
  if ! has_dir "${FRONTEND_DIR}"; then
    echo "✖ frontend/ not found at: ${FRONTEND_DIR}"
    return 1
  fi
  if ! has_cmd node || ! has_cmd npm; then
    echo "✖ node/npm is not installed (need Node.js for Vite/React)"
    return 1
  fi

  cd "${FRONTEND_DIR}"
  if [[ ! -d node_modules ]]; then
    echo "▶ Installing frontend deps (npm install)"
    npm install
  fi

  echo "▶ Running frontend (npm run dev at ${FRONTEND_URL})"
  run_and_wait npm run dev -- --host "${FRONTEND_HOST}" --port "${FRONTEND_PORT}"
}

generate_openapi_client() {
  if ! has_dir "${FRONTEND_DIR}"; then
    echo "✖ frontend/ not found at: ${FRONTEND_DIR}"
    return 1
  fi
  if ! has_cmd node || ! has_cmd npm; then
    echo "✖ node/npm is not installed (need Node.js for openapi-ts)"
    return 1
  fi
  echo "▶ Generating OpenAPI client (ensure backend is running at ${BACKEND_HOST}:${BACKEND_PORT})"
  run_and_wait_in_dir "${FRONTEND_DIR}" \
    npx @hey-api/openapi-ts -i "http://${BACKEND_HOST}:${BACKEND_PORT}/openapi.json" -o src/client
  echo "✔ OpenAPI client generated"
}

generate_postman_collection() {
  if ! has_cmd docker; then
    echo "✖ docker is not installed (needed for openapi-to-postman image)"
    return 1
  fi
  if ! has_cmd jq; then
    echo "✖ jq is not installed (needed to patch the Postman collection)"
    return 1
  fi
  echo "▶ Generating Postman collection (ensure backend is running at ${BACKEND_HOST}:${BACKEND_PORT})"
  run_and_wait_in_dir "${ROOT_DIR}" \
    bash -c \
    "docker run --rm --network host \
      --user \"$(id -u):$(id -g)\" \
      -v \"${PWD}:/local\" \
      openapitools/openapi-generator-cli sh -lc '
        curl -sS \"http://${BACKEND_HOST}:${BACKEND_PORT}/openapi.json\" -o /tmp/openapi.json &&
        openapi-generator-cli generate -i /tmp/openapi.json -g postman-collection -o /local/postman
      '"
  if [[ -f "${ROOT_DIR}/postman/patch-collection.sh" ]]; then
    echo "▶ Patching Postman collection"
    run_and_wait_in_dir "${ROOT_DIR}/postman" bash patch-collection.sh
    echo "✔ Postman collection patched"
  else
    echo "ℹ patch-collection.sh not found; skipping patch step"
  fi
  echo "✔ Postman collection generated"
}

# ---------- infra (MLflow stack) ----------
docker_compose() {
  # Prefer docker compose (v2); fall back to docker-compose (v1)
  if docker compose version >/dev/null 2>&1; then
    docker compose "$@"
  elif has_cmd docker-compose; then
    docker-compose "$@"
  else
    echo "✖ docker compose (v2) or docker-compose (v1) is required" >&2
    return 127
  fi
}

infra_compose() {
  ( cd "${INFRA_DIR}" && docker_compose "$@" )
}

infra_env_value() {
  local key="$1" default="$2" file="${INFRA_DIR}/.env" value=""
  if [[ -f "${file}" ]]; then
    value="$(grep -E "^${key}=" "${file}" | tail -n1 | cut -d= -f2- | tr -d '\r')"
  fi
  echo "${value:-${default}}"
}

ensure_infra_env() {
  if [[ -f "${INFRA_DIR}/.env" ]]; then
    return 0
  fi
  if [[ -f "${INFRA_DIR}/.env.example" ]]; then
    echo "▶ Creating ${INFRA_DIR}/.env from .env.example"
    cp "${INFRA_DIR}/.env.example" "${INFRA_DIR}/.env"
    return 0
  fi
  echo "✖ ${INFRA_DIR}/.env not found and no .env.example to copy"
  return 1
}

check_infra_prereqs() {
  if ! has_dir "${INFRA_DIR}"; then
    echo "✖ infra dir not found at: ${INFRA_DIR}"
    return 1
  fi
  if ! has_cmd docker; then
    echo "✖ docker is not installed (needed to run the MLflow infra)"
    return 1
  fi
}

run_infra_up() {
  check_infra_prereqs || return 1
  ensure_infra_env || return 1

  echo "▶ Starting MLflow infra (docker compose up -d) in ${INFRA_DIR}"
  if ! infra_compose up -d; then
    echo "✖ Failed to start MLflow infra"
    return 1
  fi

  local mlflow_port minio_ui_port
  mlflow_port="$(infra_env_value MLFLOW_PORT 5050)"
  minio_ui_port="$(infra_env_value S3_UI_PORT 9001)"
  echo "✔ MLflow infra started"
  echo "  MLflow UI:     http://localhost:${mlflow_port}"
  echo "  MinIO console: http://localhost:${minio_ui_port}"
}

run_infra_down() {
  check_infra_prereqs || return 1

  echo "▶ Stopping MLflow infra (docker compose down) in ${INFRA_DIR}"
  if ! infra_compose down; then
    echo "✖ Failed to stop MLflow infra"
    return 1
  fi
  echo "✔ MLflow infra stopped"
}

install_all() {
  echo "▶ Installing all dependencies (python + frontend)..."

  install_python_all >/dev/null

  if has_dir "${FRONTEND_DIR}"; then
    if ! has_cmd node || ! has_cmd npm; then
      echo "✖ node/npm is not installed (needed for Vite). Install Node.js and retry."
      return 1
    fi
    echo "▶ Installing frontend deps (npm install)"
    ( cd "${FRONTEND_DIR}" && npm install )
    echo "✔ frontend deps installed"
  else
    echo "ℹ frontend/ not found, skipping"
  fi

  echo "✔ All dependencies installed"
}

# ---------- misc ----------
show_tree() {
  echo "▶ tree (filtered)"
  if has_cmd tree; then
    tree -L 4 -I "node_modules|.venv|__pycache__|*.egg-info|build|dist|*.db|.pytest_cache|.mypy_cache"
  else
    echo "✖ tree is not installed. Install: sudo apt install tree"
  fi
}

status_info() {
  echo "ROOT:      ${ROOT_DIR}"
  echo "VENV:      ${VENV_DIR} (exists: $([[ -d "${VENV_DIR}" ]] && echo yes || echo no))"
  echo "SERVICE:   ${SERVICE_DIR} (exists: $(has_dir "${SERVICE_DIR}" && echo yes || echo no))"
  echo "FRONTEND:  ${FRONTEND_DIR} (exists: $(has_dir "${FRONTEND_DIR}" && echo yes || echo no))"
  if has_cmd uv; then
    echo "UV:        $(uv_cmd --version)"
  else
    echo "UV:        uv not found"
  fi
  if [[ -x "${VENV_DIR}/bin/python" ]]; then
    echo "PYTHON:    $("${VENV_DIR}/bin/python" -V)"
  else
    echo "PYTHON:    venv not created"
  fi
}

# ---------- logs viewer (interactive, like the main menu) ----------
LOGS_MENU_OPTIONS=()
LOGS_MENU_SELECTED=0
LOGS_MENU_RESULT=""

logs_menu_up() {
  local size="$1"
  if [[ "${LOGS_MENU_SELECTED}" -le 0 ]]; then
    LOGS_MENU_SELECTED=$((size - 1))
  else
    LOGS_MENU_SELECTED=$((LOGS_MENU_SELECTED - 1))
  fi
}

logs_menu_down() {
  local size="$1"
  if [[ "${LOGS_MENU_SELECTED}" -ge $((size - 1)) ]]; then
    LOGS_MENU_SELECTED=0
  else
    LOGS_MENU_SELECTED=$((LOGS_MENU_SELECTED + 1))
  fi
}

logs_menu_handle_input() {
  local input="$1" size="$2" key
  LOGS_MENU_RESULT=""

  while [[ -n "${input}" ]]; do
    if [[ "${input}" == $'\e[A'* ]]; then
      logs_menu_up "${size}"; input="${input:3}"; continue
    fi
    if [[ "${input}" == $'\e[B'* ]]; then
      logs_menu_down "${size}"; input="${input:3}"; continue
    fi
    if [[ "${input}" == $'\e[C'* || "${input}" == $'\e[D'* ]]; then
      input="${input:3}"; continue
    fi

    key="${input:0:1}"
    input="${input:1}"

    case "${key}" in
      "k"|"K") logs_menu_up "${size}" ;;
      "j"|"J") logs_menu_down "${size}" ;;
      $'\n'|$'\r'|" ") LOGS_MENU_RESULT="select"; return 0 ;;
      "q"|"Q"|"b"|"B"|$'\e') LOGS_MENU_RESULT="back"; return 0 ;;
    esac
  done
}

render_logs_menu() {
  local tail_lines="$1"
  local width i marker label f

  width="$(tput cols 2>/dev/null || echo 80)"
  (( width < 80 )) && width=80

  {
    printf '\033[H'
    fast_window_border "${width}"
    fast_window_title "Service logs" "${width}"
    fast_window_border "${width}"
    fast_window_line "Up/Down or j/k: move    Enter/Space: view    q/Esc/b: back" "${width}"
    fast_window_line "Tail: last ${tail_lines} lines    Dir: ${BOOTSTRAP_LOG_DIR}" "${width}"
    fast_window_border "${width}"

    for i in "${!LOGS_MENU_OPTIONS[@]}"; do
      if [[ "${LOGS_MENU_OPTIONS[$i]}" == "all" ]]; then
        label="all logs (backend/frontend/jupyter/mlflow)"
      else
        f="$(managed_log_file "${LOGS_MENU_OPTIONS[$i]}")"
        label="$(printf '%-9s %s' "${LOGS_MENU_OPTIONS[$i]}" "${f}")"
      fi
      if [[ "${i}" -eq "${LOGS_MENU_SELECTED}" ]]; then
        marker="> [*]"
      else
        marker="  [ ]"
      fi
      fast_window_line "${marker} ${label}" "${width}"
    done
    fast_window_border "${width}"
    printf '\033[J'
  }
}

logs_show_tail() {
  local tail_lines="$1"
  shift
  local files=("$@")

  # Drop back to the normal screen (with scrollback) to show the log
  dashboard_leave_screen
  printf '\033[H\033[J'
  if [[ "${#files[@]}" -eq 1 ]]; then
    echo "▶ ${files[0]} (last ${tail_lines} lines)"
  else
    echo "▶ All logs (last ${tail_lines} lines each)"
  fi
  echo
  tail -n "${tail_lines}" "${files[@]}"
  echo
  pause
  dashboard_enter_screen
}

logs_show_message() {
  dashboard_leave_screen
  printf '\033[H\033[J'
  local line
  for line in "$@"; do
    echo "${line}"
  done
  echo
  pause
  dashboard_enter_screen
}

view_logs() {
  ensure_bootstrap_state
  local tail_lines="${LOG_TAIL_LINES:-200}"
  local names=() name f dirty choice
  local files=()

  for name in backend frontend jupyter mlflow; do
    f="$(managed_log_file "${name}")"
    [[ -f "${f}" ]] && names+=("${name}")
  done

  if [[ "${#names[@]}" -eq 0 ]]; then
    logs_show_message \
      "No log files yet in ${BOOTSTRAP_LOG_DIR}" \
      "Start a service (backend/frontend/jupyter/mlflow) to generate logs."
    DASH_MESSAGE="View logs: no logs yet"
    return 0
  fi

  LOGS_MENU_OPTIONS=("${names[@]}" "all")
  LOGS_MENU_SELECTED=0
  dirty=1

  while true; do
    if [[ "${dirty}" -eq 1 ]]; then
      render_logs_menu "${tail_lines}"
      dirty=0
    fi

    if read_dashboard_input; then
      logs_menu_handle_input "${DASH_INPUT}" "${#LOGS_MENU_OPTIONS[@]}"
      case "${LOGS_MENU_RESULT}" in
        select)
          choice="${LOGS_MENU_OPTIONS[$LOGS_MENU_SELECTED]}"
          files=()
          if [[ "${choice}" == "all" ]]; then
            for name in "${names[@]}"; do
              files+=("$(managed_log_file "${name}")")
            done
          else
            files+=("$(managed_log_file "${choice}")")
          fi
          logs_show_tail "${tail_lines}" "${files[@]}"
          ;;
        back)
          break
          ;;
      esac
      dirty=1
    fi
  done

  DASH_MESSAGE="View logs: closed"
}

clean_deps() {
  echo "⚠️  This will remove dependency artifacts:"
  echo "   - Python venv (.venv)"
  echo "   - Python build artifacts (build/, dist/, *.egg-info, __pycache__)"
  echo "   - Frontend node_modules/"
  echo
  read -r -p "Are you sure? [y/N]: " CONFIRM

  if [[ ! "${CONFIRM}" =~ ^[Yy]$ ]]; then
    echo "✖ Cancelled"
    return 0
  fi

  echo "▶ Cleaning Python artifacts..."
  rm -rf "${VENV_DIR}"

  find "${ROOT_DIR}" -type d -name "__pycache__" -prune -exec rm -rf {} +
  find "${ROOT_DIR}" -type d -name "*.egg-info" -prune -exec rm -rf {} +

  rm -rf "${ROOT_DIR}/build" "${ROOT_DIR}/dist"

  echo "▶ Cleaning frontend artifacts..."
  if has_dir "${FRONTEND_DIR}/node_modules"; then
    rm -rf "${FRONTEND_DIR}/node_modules"
  fi

  echo "✔ Dependencies cleaned"
}

# ---------- menu ----------
MENU_ACTIONS=(
  "toggle_backend"
  "toggle_frontend"
  "toggle_jupyter"
  "start_all_services"
  "stop_all_services"
  "infra_up"
  "infra_down"
  "uv_setup"
  "install_base"
  "install_notebooks"
  "install_dev"
  "install_all"
  "alembic_upgrade"
  "alembic_revision"
  "generate_openapi_client"
  "generate_postman_collection"
  "show_tree"
  "status_info"
  "view_logs"
  "clean_deps"
  "publish_models_to_wandb"
  "download_best_model"
  "upload_model_to_mlflow"
  "exit"
)

MENU_SELECTED=0
DASH_MESSAGE="Ready"
DASH_FORCE_STATUS_REFRESH=1
BACKEND_ACTIVE=0
FRONTEND_ACTIVE=0
JUPYTER_ACTIVE=0
BACKEND_STATUS_TEXT="..."
FRONTEND_STATUS_TEXT="..."
JUPYTER_STATUS_TEXT="..."
INFRA_STATUS_TEXT="..."
VENV_STATUS_TEXT="..."
INFRA_MLFLOW_PORT="5050"
DASHBOARD_SCREEN_ACTIVE=0

ensure_bootstrap_state() {
  mkdir -p "${BOOTSTRAP_PID_DIR}" "${BOOTSTRAP_LOG_DIR}"
}

managed_pid_file() {
  echo "${BOOTSTRAP_PID_DIR}/$1.pid"
}

managed_log_file() {
  echo "${BOOTSTRAP_LOG_DIR}/$1.log"
}

managed_pid() {
  local file
  file="$(managed_pid_file "$1")"
  if [[ -f "${file}" ]]; then
    tr -d '[:space:]' < "${file}" || true
  fi
  return 0
}

pid_running() {
  local pid="$1"
  [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1
}

managed_running() {
  local name="$1" pid file
  file="$(managed_pid_file "${name}")"
  pid="$(managed_pid "${name}")"

  if pid_running "${pid}"; then
    return 0
  fi

  [[ -f "${file}" ]] && rm -f "${file}"
  return 1
}

service_port() {
  case "$1" in
    backend) echo "${BACKEND_PORT}" ;;
    frontend) echo "${FRONTEND_PORT}" ;;
    jupyter) echo "${JUPYTER_PORT}" ;;
  esac
}

port_pids() {
  local port="$1"

  {
    if has_cmd ss; then
      ss -ltnp "sport = :${port}" 2>/dev/null | sed -nE 's/.*pid=([0-9]+).*/\1/p' || true
    fi
    if has_cmd fuser; then
      fuser -n tcp "${port}" 2>/dev/null | tr ' ' '\n' || true
    fi
    if has_cmd lsof; then
      lsof -tiTCP:"${port}" -sTCP:LISTEN 2>/dev/null || true
    fi
    if has_cmd netstat; then
      netstat -tulpn 2>/dev/null | awk -v port=":${port}" '$4 ~ port"$" && $6 == "LISTEN" { split($7, a, "/"); print a[1] }' || true
    fi
  } | awk '/^[0-9]+$/ { print }' | sort -u
}

port_listening() {
  local port="$1"

  if has_cmd ss; then
    ss -ltn "sport = :${port}" 2>/dev/null | awk 'NR > 1 { found=1 } END { exit !found }'
    return $?
  fi

  [[ -n "$(port_pids "${port}")" ]]
}

service_port_pids() {
  local port
  port="$(service_port "$1" || true)"
  [[ -z "${port}" ]] && return 0
  port_pids "${port}" || true
}

service_port_listening() {
  local port
  port="$(service_port "$1" || true)"
  [[ -n "${port}" ]] && port_listening "${port}"
}

service_running() {
  local name="$1"
  managed_running "${name}" && return 0
  service_port_listening "${name}"
}

service_status() {
  local name="$1" label="$2" pid port_pid
  pid="$(managed_pid "${name}" || true)"

  if managed_running "${name}"; then
    echo "RUN ${label} pid=${pid}"
  elif port_pid="$(service_port_pids "${name}")" && [[ -n "${port_pid}" ]]; then
    echo "RUN ${label} pid=${port_pid%%$'\n'*}"
  elif service_port_listening "${name}"; then
    echo "RUN ${label} pid=unknown"
  else
    echo "STOP ${label}"
  fi
}

refresh_service_status_cache() {
  BACKEND_STATUS_TEXT="$(service_status backend ":${BACKEND_PORT}")"
  FRONTEND_STATUS_TEXT="$(service_status frontend ":${FRONTEND_PORT}")"
  JUPYTER_STATUS_TEXT="$(service_status jupyter ":${JUPYTER_PORT}")"
  INFRA_MLFLOW_PORT="$(infra_env_value MLFLOW_PORT 5050)"
  INFRA_STATUS_TEXT="$(infra_status "${INFRA_MLFLOW_PORT}")"
  VENV_STATUS_TEXT="$(venv_status)"

  [[ "${BACKEND_STATUS_TEXT}" == RUN* ]] && BACKEND_ACTIVE=1 || BACKEND_ACTIVE=0
  [[ "${FRONTEND_STATUS_TEXT}" == RUN* ]] && FRONTEND_ACTIVE=1 || FRONTEND_ACTIVE=0
  [[ "${JUPYTER_STATUS_TEXT}" == RUN* ]] && JUPYTER_ACTIVE=1 || JUPYTER_ACTIVE=0
}

cached_service_running() {
  case "$1" in
    backend) [[ "${BACKEND_ACTIVE}" -eq 1 ]] ;;
    frontend) [[ "${FRONTEND_ACTIVE}" -eq 1 ]] ;;
    jupyter) [[ "${JUPYTER_ACTIVE}" -eq 1 ]] ;;
    *) return 1 ;;
  esac
}

venv_status() {
  if [[ -x "${VENV_DIR}/bin/python" ]]; then
    echo "READY $("${VENV_DIR}/bin/python" -V 2>/dev/null | awk '{print $2}')"
  elif [[ -d "${VENV_DIR}" ]]; then
    echo "BROKEN"
  else
    echo "MISSING"
  fi
}

infra_status() {
  local port="$1"
  if [[ -n "${port}" ]] && port_listening "${port}"; then
    echo "RUN :${port}"
  else
    echo "STOP"
  fi
}

kill_process_tree() {
  local pid="$1" child

  if has_cmd pgrep; then
    while read -r child; do
      [[ -n "${child}" ]] && kill_process_tree "${child}"
    done < <(pgrep -P "${pid}" 2>/dev/null || true)
  fi

  kill "${pid}" >/dev/null 2>&1 || true
}

stop_managed_service() {
  local name="$1" pid file waited port_pid pid_to_kill remaining_pids stopped port
  file="$(managed_pid_file "${name}")"
  pid="$(managed_pid "${name}" || true)"
  port="$(service_port "${name}" || true)"
  stopped=0

  if pid_running "${pid}"; then
    kill_process_tree "${pid}"
    stopped=1
  fi

  while read -r port_pid; do
    [[ -z "${port_pid}" ]] && continue
    [[ "${port_pid}" == "${pid}" ]] && continue
    kill_process_tree "${port_pid}"
    stopped=1
  done < <(service_port_pids "${name}" || true)

  waited=0
  while [[ "${waited}" -lt 20 ]]; do
    if ! service_port_listening "${name}" && ! pid_running "${pid}"; then
      break
    fi
    sleep 0.1
    waited=$((waited + 1))
  done

  if pid_running "${pid}"; then
    kill -9 "${pid}" >/dev/null 2>&1 || true
  fi
  while read -r pid_to_kill; do
    [[ -n "${pid_to_kill}" ]] && kill -9 "${pid_to_kill}" >/dev/null 2>&1 || true
  done < <(service_port_pids "${name}" || true)

  rm -f "${file}"

  remaining_pids="$(service_port_pids "${name}" || true)"
  if service_port_listening "${name}"; then
    if [[ -n "${remaining_pids}" ]]; then
      DASH_MESSAGE="${name}: cannot stop; port ${port} still busy, pid ${remaining_pids%%$'\n'*}"
    else
      DASH_MESSAGE="${name}: cannot stop; port ${port} still busy"
    fi
  elif [[ "${stopped}" -eq 1 ]]; then
    DASH_MESSAGE="${name}: stopped"
  else
    DASH_MESSAGE="${name}: already stopped"
  fi
}

start_backend_managed() {
  ensure_bootstrap_state
  if service_running "backend"; then
    DASH_MESSAGE="backend: already running"
    return 0
  fi

  (
    set -euo pipefail
    trap '' HUP
    install_python_base
    cd "${SERVICE_DIR}"
    uv_run python -m uvicorn "${BACKEND_MODULE}" --reload --host "${BACKEND_HOST}" --port "${BACKEND_PORT}"
  ) >"$(managed_log_file backend)" 2>&1 < /dev/null &

  local pid=$!
  echo "${pid}" > "$(managed_pid_file backend)"
  disown "${pid}" >/dev/null 2>&1 || true
  DASH_MESSAGE="backend: starting, log $(managed_log_file backend)"
}

start_frontend_managed() {
  ensure_bootstrap_state
  if service_running "frontend"; then
    DASH_MESSAGE="frontend: already running"
    return 0
  fi
  if ! has_dir "${FRONTEND_DIR}"; then
    DASH_MESSAGE="frontend: directory not found"
    return 0
  fi
  if ! has_cmd node || ! has_cmd npm; then
    DASH_MESSAGE="frontend: node/npm is not installed"
    return 0
  fi

  (
    set -euo pipefail
    trap '' HUP
    cd "${FRONTEND_DIR}"
    if [[ ! -d node_modules ]]; then
      npm install
    fi
    npm run dev -- --host "${FRONTEND_HOST}" --port "${FRONTEND_PORT}" --strictPort
  ) >"$(managed_log_file frontend)" 2>&1 < /dev/null &

  local pid=$!
  echo "${pid}" > "$(managed_pid_file frontend)"
  disown "${pid}" >/dev/null 2>&1 || true
  DASH_MESSAGE="frontend: starting, log $(managed_log_file frontend)"
}

start_jupyter_managed() {
  ensure_bootstrap_state
  if service_running "jupyter"; then
    DASH_MESSAGE="jupyter: already running"
    return 0
  fi

  (
    set -euo pipefail
    trap '' HUP
    install_python_notebooks
    cd "${ROOT_DIR}"
    uv_run jupyter-lab --no-browser --ip="${JUPYTER_HOST}" --port="${JUPYTER_PORT}" --ServerApp.port_retries=0
  ) >"$(managed_log_file jupyter)" 2>&1 < /dev/null &

  local pid=$!
  echo "${pid}" > "$(managed_pid_file jupyter)"
  disown "${pid}" >/dev/null 2>&1 || true
  DASH_MESSAGE="jupyter: starting, log $(managed_log_file jupyter)"
}

toggle_managed_service() {
  local name="$1"
  if service_running "${name}"; then
    stop_managed_service "${name}"
    return 0
  fi

  case "${name}" in
    backend) start_backend_managed ;;
    frontend) start_frontend_managed ;;
    jupyter) start_jupyter_managed ;;
  esac
}

start_infra_managed() {
  ensure_bootstrap_state
  if ! has_dir "${INFRA_DIR}"; then
    DASH_MESSAGE="mlflow: infra dir not found"
    return 0
  fi
  if ! has_cmd docker; then
    DASH_MESSAGE="mlflow: docker is not installed"
    return 0
  fi

  (
    set -euo pipefail
    trap '' HUP
    run_infra_up
  ) >"$(managed_log_file mlflow)" 2>&1 < /dev/null &

  local pid=$!
  disown "${pid}" >/dev/null 2>&1 || true
  DASH_MESSAGE="mlflow: starting, log $(managed_log_file mlflow)"
}

stop_infra_managed() {
  if ! has_dir "${INFRA_DIR}" || ! has_cmd docker; then
    DASH_MESSAGE="mlflow: already stopped"
    return 0
  fi

  (
    set -euo pipefail
    trap '' HUP
    run_infra_down
  ) >"$(managed_log_file mlflow)" 2>&1 < /dev/null &

  local pid=$!
  disown "${pid}" >/dev/null 2>&1 || true
  DASH_MESSAGE="mlflow: stopping, log $(managed_log_file mlflow)"
}

start_all_managed_services() {
  start_backend_managed
  start_frontend_managed
  start_jupyter_managed
  start_infra_managed
  DASH_MESSAGE="backend/frontend/jupyter/mlflow: start requested"
}

stop_all_managed_services() {
  local backend_msg frontend_msg jupyter_msg mlflow_msg

  stop_managed_service "backend"
  backend_msg="${DASH_MESSAGE}"
  stop_managed_service "frontend"
  frontend_msg="${DASH_MESSAGE}"
  stop_managed_service "jupyter"
  jupyter_msg="${DASH_MESSAGE}"
  stop_infra_managed
  mlflow_msg="${DASH_MESSAGE}"
  DASH_MESSAGE="stop all: ${backend_msg}; ${frontend_msg}; ${jupyter_msg}; ${mlflow_msg}"
}

menu_label() {
  local action="$1"
  case "${action}" in
    toggle_backend)
      cached_service_running "backend" && echo "🚀 Stop backend (${BACKEND_URL}, port ${BACKEND_PORT})" || echo "🚀 Start backend (${BACKEND_URL}, port ${BACKEND_PORT})"
      ;;
    toggle_frontend)
      cached_service_running "frontend" && echo "🎨 Stop frontend (${FRONTEND_URL})" || echo "🎨 Start frontend (${FRONTEND_URL})"
      ;;
    toggle_jupyter)
      cached_service_running "jupyter" && echo "📓 Stop JupyterLab (${JUPYTER_URL})" || echo "📓 Start JupyterLab (${JUPYTER_URL})"
      ;;
    start_all_services) echo "🟢 Start backend + frontend + JupyterLab + MLflow" ;;
    stop_all_services) echo "🔴 Stop backend + frontend + JupyterLab + MLflow" ;;
    infra_up) echo "🐳 Start MLflow infra (docker compose up -d)" ;;
    infra_down) echo "⛔ Stop MLflow infra (docker compose down)" ;;
    uv_setup) echo "🐍 Setup venv (uv venv)" ;;
    install_base) echo "📦 Install python deps (editable)" ;;
    install_notebooks) echo "📚 Install python notebook extras" ;;
    install_dev) echo "🔧 Install python dev extras" ;;
    install_all) echo "🧰 Install ALL deps (python + frontend)" ;;
    alembic_upgrade) echo "🆙 Alembic upgrade head" ;;
    alembic_revision) echo "📝 Alembic revision --autogenerate" ;;
    generate_openapi_client) echo "🔗 Generate OpenAPI client (frontend)" ;;
    generate_postman_collection) echo "📮 Generate Postman collection" ;;
    show_tree) echo "🌳 Show tree (filtered)" ;;
    status_info) echo "📊 Status details" ;;
    view_logs) echo "📜 View service logs (backend/frontend/jupyter/mlflow)" ;;
    clean_deps) echo "🧹 Clean dependencies (venv/build/node_modules)" ;;
    publish_models_to_wandb) echo "📤 Publish models to W&B" ;;
    download_best_model) echo "📥 Download best model from MLflow" ;;
    upload_model_to_mlflow) echo "📌 Register model file in MLflow (env name+alias)" ;;
    exit) echo "🚪 Exit menu (services keep running)" ;;
  esac
}

dashboard_run_blocking() {
  local title="$1" status
  shift

  dashboard_leave_screen
  printf '\033[H\033[J'
  echo "▶ ${title}"
  echo

  set +e
  "$@"
  status=$?
  set -e

  echo
  if [[ "${status}" -eq 0 ]]; then
    echo "✔ ${title} finished"
    DASH_MESSAGE="${title}: ok"
  else
    echo "✖ ${title} failed with exit code ${status}"
    DASH_MESSAGE="${title}: failed (${status})"
  fi
  pause

  dashboard_enter_screen
}

dashboard_exit() {
  dashboard_leave_screen
  printf '\033[H\033[J'
  echo "Bye!"
  echo "Managed services keep running until you stop them from this menu or kill their PIDs."
  echo "Logs: ${BOOTSTRAP_LOG_DIR}"
  exit 0
}

menu_up() {
  local size="${#MENU_ACTIONS[@]}"
  if [[ "${MENU_SELECTED}" -le 0 ]]; then
    MENU_SELECTED=$((size - 1))
  else
    MENU_SELECTED=$((MENU_SELECTED - 1))
  fi
}

menu_down() {
  local size="${#MENU_ACTIONS[@]}"
  if [[ "${MENU_SELECTED}" -ge $((size - 1)) ]]; then
    MENU_SELECTED=0
  else
    MENU_SELECTED=$((MENU_SELECTED + 1))
  fi
}

run_selected_menu_action() {
  local action="${MENU_ACTIONS[$MENU_SELECTED]}"
  case "${action}" in
    toggle_backend) toggle_managed_service "backend" ;;
    toggle_frontend) toggle_managed_service "frontend" ;;
    toggle_jupyter) toggle_managed_service "jupyter" ;;
    start_all_services) start_all_managed_services ;;
    stop_all_services) stop_all_managed_services ;;
    infra_up) dashboard_run_blocking "Start MLflow infra" run_infra_up ;;
    infra_down) dashboard_run_blocking "Stop MLflow infra" run_infra_down ;;
    uv_setup) dashboard_run_blocking "Setup venv" uv_in_venv ;;
    install_base) dashboard_run_blocking "Install python deps" install_python_base ;;
    install_notebooks) dashboard_run_blocking "Install python notebook extras" install_python_notebooks ;;
    install_dev) dashboard_run_blocking "Install python dev extras" install_python_dev ;;
    install_all) dashboard_run_blocking "Install all dependencies" install_all ;;
    alembic_upgrade) dashboard_run_blocking "Alembic upgrade head" alembic_upgrade ;;
    alembic_revision) dashboard_run_blocking "Alembic revision" alembic_revision ;;
    generate_openapi_client) dashboard_run_blocking "Generate OpenAPI client" generate_openapi_client ;;
    generate_postman_collection) dashboard_run_blocking "Generate Postman collection" generate_postman_collection ;;
    show_tree) dashboard_run_blocking "Show tree" show_tree ;;
    status_info) dashboard_run_blocking "Status details" status_info ;;
    view_logs) view_logs ;;
    clean_deps) dashboard_run_blocking "Clean dependencies" clean_deps ;;
    publish_models_to_wandb) dashboard_run_blocking "Publish models to W&B" publish_models_to_wandb ;;
    download_best_model) dashboard_run_blocking "Download best model" download_best_model ;;
    upload_model_to_mlflow) dashboard_run_blocking "Register model in MLflow" upload_model_to_mlflow ;;
    exit) dashboard_exit ;;
  esac
  DASH_FORCE_STATUS_REFRESH=1
}

handle_dashboard_input() {
  local input="$1"
  local key

  while [[ -n "${input}" ]]; do
    if [[ "${input}" == $'\e[A'* ]]; then
      menu_up
      input="${input:3}"
      continue
    fi
    if [[ "${input}" == $'\e[B'* ]]; then
      menu_down
      input="${input:3}"
      continue
    fi

    key="${input:0:1}"
    input="${input:1}"

    case "${key}" in
      "k"|"K") menu_up ;;
      "j"|"J") menu_down ;;
      $'\n'|$'\r'|" ") run_selected_menu_action; return 0 ;;
      "q"|"Q"|$'\e') dashboard_exit ;;
    esac
  done
}

fit_text() {
  local text="$1" width="$2"
  if (( ${#text} > width )); then
    printf '%s' "${text:0:width}"
  else
    printf '%-*s' "${width}" "${text}"
  fi
}

fast_window_line() {
  local text="$1" width="$2"
  printf '|%s|\n' "$(fit_text "${text}" "$((width - 2))")"
}

fast_window_border() {
  local width="$1"
  printf '+'
  printf '%*s' "$((width - 2))" '' | tr ' ' '-'
  printf '+\n'
}

fast_window_title() {
  local title="$1" width="$2" inner padding_left padding_right
  inner=$((width - 2))
  if (( ${#title} > inner )); then
    title="${title:0:inner}"
  fi
  padding_left=$(((inner - ${#title}) / 2))
  padding_right=$((inner - padding_left - ${#title}))
  printf '|%*s%s%*s|\n' "${padding_left}" '' "${title}" "${padding_right}" ''
}

fast_status_row() {
  local width="$1"
  shift
  local out="" cell first=1 value
  for value in "$@"; do
    if (( first )); then
      first=0
    else
      out+=" "
    fi
    printf -v cell '%-*.*s' "${width}" "${width}" "${value}"
    out+="${cell}"
  done
  printf '%s' "${out}"
}

fast_dashboard_render() {
  local width action label marker i selected line status_colw

  width="$(tput cols 2>/dev/null || echo 80)"
  (( width < 80 )) && width=80

  status_colw=$(( (width - 6) / 5 ))
  (( status_colw < 9 )) && status_colw=9
  (( status_colw > 17 )) && status_colw=17

  {
    printf '\033[H'

    fast_window_border "${width}"
    fast_window_title "Project2025 status" "${width}"
    fast_window_border "${width}"
    fast_window_line "$(fast_status_row "${status_colw}" "Backend" "UI" "Jupyter" "MLflow" "Venv")" "${width}"
    fast_window_line "$(fast_status_row "${status_colw}" "${BACKEND_STATUS_TEXT}" "${FRONTEND_STATUS_TEXT}" "${JUPYTER_STATUS_TEXT}" "${INFRA_STATUS_TEXT}" "${VENV_STATUS_TEXT}")" "${width}"
    fast_window_border "${width}"
    fast_window_line "Backend API: ${BACKEND_URL}    port: ${BACKEND_PORT}" "${width}"
    fast_window_line "Frontend UI: ${FRONTEND_URL}" "${width}"
    fast_window_line "JupyterLab:  ${JUPYTER_URL}" "${width}"
    fast_window_line "MLflow UI:   http://localhost:${INFRA_MLFLOW_PORT}" "${width}"
    fast_window_border "${width}"

    fast_window_border "${width}"
    fast_window_title "Actions" "${width}"
    fast_window_border "${width}"
    fast_window_line "Up/Down or j/k: move    Enter/Space: run    q/Esc: exit" "${width}"
    fast_window_border "${width}"

    for i in "${!MENU_ACTIONS[@]}"; do
      action="${MENU_ACTIONS[$i]}"
      label="$(menu_label "${action}")"
      if [[ "${i}" -eq "${MENU_SELECTED}" ]]; then
        marker="> [*]"
      else
        marker="  [ ]"
      fi
      fast_window_line "${marker} ${label}" "${width}"
    done
    fast_window_border "${width}"

    fast_window_border "${width}"
    fast_window_title "Message" "${width}"
    fast_window_border "${width}"
    fast_window_line "${DASH_MESSAGE}" "${width}"
    fast_window_line "Logs: ${BOOTSTRAP_LOG_DIR}/backend.log | frontend.log | jupyter.log" "${width}"
    fast_window_border "${width}"
    printf '\033[J'
  }
}

dashboard_status_snapshot() {
  printf '%s|%s|%s|%s|%s' \
    "${BACKEND_STATUS_TEXT}" \
    "${FRONTEND_STATUS_TEXT}" \
    "${JUPYTER_STATUS_TEXT}" \
    "${INFRA_STATUS_TEXT}" \
    "${VENV_STATUS_TEXT}"
}

restore_terminal() {
  printf '\033[0m'
  tput cnorm 2>/dev/null || true
  stty sane 2>/dev/null || stty icanon echo 2>/dev/null || true
}

dashboard_cleanup() {
  restore_terminal
  if [[ "${DASHBOARD_SCREEN_ACTIVE:-0}" -eq 1 ]]; then
    tput rmcup 2>/dev/null || true
    DASHBOARD_SCREEN_ACTIVE=0
  fi
}

dashboard_enter_screen() {
  tput smcup 2>/dev/null || true
  DASHBOARD_SCREEN_ACTIVE=1
  stty -icanon -echo min 0 time 0 2>/dev/null || true
  tput civis 2>/dev/null || true
}

dashboard_leave_screen() {
  dashboard_cleanup
}

read_dashboard_input() {
  local key
  DASH_INPUT=""

  if ! IFS= read -rsN1 -t 0.02 key; then
    return 1
  fi

  DASH_INPUT="${key}"
  while IFS= read -rsN1 -t 0.01 key; do
    DASH_INPUT+="${key}"
  done
  return 0
}

dashboard_loop() {
  local last_status now dirty status_snapshot last_status_snapshot

  dashboard_enter_screen
  last_status=0
  last_status_snapshot=""
  dirty=1
  refresh_service_status_cache

  while true; do
    if [[ "${DASH_FORCE_STATUS_REFRESH}" -eq 1 ]]; then
      refresh_service_status_cache
      DASH_FORCE_STATUS_REFRESH=0
      dirty=1
    fi

    if [[ "${dirty}" -eq 1 ]]; then
      fast_dashboard_render
      dirty=0
      printf -v last_status '%(%s)T' -1
      last_status_snapshot="$(dashboard_status_snapshot)"
    fi

    if read_dashboard_input; then
      handle_dashboard_input "${DASH_INPUT}"
      dirty=1
    fi

    printf -v now '%(%s)T' -1
    if [[ $((now - last_status)) -ge PROCESS_POLL_INTERVAL ]]; then
      refresh_service_status_cache
      status_snapshot="$(dashboard_status_snapshot)"
      if [[ "${status_snapshot}" != "${last_status_snapshot}" ]]; then
        dirty=1
      else
        last_status="${now}"
      fi
    fi
  done
}

ensure_bootstrap_state
trap 'dashboard_cleanup' EXIT
trap 'dashboard_cleanup; exit 130' INT TERM
dashboard_loop
