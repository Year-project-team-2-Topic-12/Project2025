#!/usr/bin/env bash
# All of this file is ai-assisted
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

SERVICE_DIR="${ROOT_DIR}/service"
FRONTEND_DIR="${ROOT_DIR}/frontend"

BACKEND_MODULE="backend.main:app"
BACKEND_HOST="127.0.0.1"
BACKEND_PORT="8000"

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
    uv_run jupyter-lab --no-browser --ip=127.0.0.1 --port=8888
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

  echo "▶ Running frontend (npm run dev)"
  run_and_wait npm run dev
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
while true; do
  echo
  echo "================= DEV MENU ================="
  echo "1) Setup venv (uv venv)"
  echo "2) Install python deps (editable)"
  echo "3) Install python dev extras"
  echo "4) Install ALL deps (python + frontend)"
  echo "5) Run backend (uvicorn --reload)"
  echo "6) Alembic upgrade head"
  echo "7) Alembic revision --autogenerate"
  echo "8) Run frontend (Vite)"
  echo "9) Generate OpenAPI client (frontend)"
  echo "10) Generate Postman collection (frontend)"
  echo "11) Run JupyterLab"
  echo "12) Show tree (filtered)"
  echo "13) Status"
  echo "14) Clean dependencies (remove venv, build artifacts, node_modules)"
  echo "0) Exit"
  echo "==========================================="
  read -r -p "Choose: " CH

  case "${CH}" in
    1) uv_in_venv; echo "✔ venv ready"; pause ;;
    2) install_python_base; pause ;;
    3) install_python_dev; pause ;;
    4) install_all; pause ;;
    5) run_backend; pause ;;
    6) alembic_upgrade; pause ;;
    7) alembic_revision; pause ;;
    8) run_frontend; pause ;;
    9) generate_openapi_client; pause ;;
    10) generate_postman_collection; pause ;;
    11) run_jupyter; pause ;;
    12) show_tree; pause ;;
    13) status_info; pause ;;
    14) clean_deps; pause ;;
    0) echo "Bye!"; exit 0 ;;
    *) echo "Unknown option: ${CH}"; pause ;;
  esac
done
