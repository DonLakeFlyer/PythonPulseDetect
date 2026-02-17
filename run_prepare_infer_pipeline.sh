#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   Positional (backward compatible):
#     ./run_prepare_infer_pipeline.sh [config] [model] [data_dir] [out_dir] [criterion] [sessions_dir]
#
#   Named options:
#     ./run_prepare_infer_pipeline.sh \
#       --config capture_config.json \
#       --data-dir artifacts/processed/cnn_simple \
#       --output-dir artifacts/models/cnn_simple \
#       --criterion f1 \
#       [--training-sessions-dir artifacts/training-sessions] [--val-frac 0.2] [--max-per-class 0] [--seed 42] [--skip-train]

print_usage() {
  cat <<'EOF'
Usage:
  Positional (backward compatible):
    ./run_prepare_infer_pipeline.sh [config] [model] [data_dir] [out_dir] [criterion] [sessions_dir]

  Named options:
    ./run_prepare_infer_pipeline.sh \
      --config capture_config.json \
      --data-dir artifacts/processed/cnn_simple \
      --output-dir artifacts/models/cnn_simple \
      --criterion f1 \
      [--training-sessions-dir artifacts/training-sessions] [--val-frac 0.2] [--max-per-class 0] [--seed 42] [--skip-train]

Options:
  --config FILE         Base capture config JSON (default: capture_config.json)
  --model FILE          Model checkpoint path (default: artifacts/models/cnn_simple/best_model.pt)
  --data-dir DIR        Processed dataset directory with X_val/y_val (default: artifacts/processed/cnn_simple)
  --output-dir DIR      Output directory for threshold/runtime files (default: artifacts/models/cnn_simple)
  --criterion NAME      Threshold criterion: f1|target_recall|target_precision (default: f1)
  --training-sessions-dir DIR
                        Parent folder of raw capture sessions (default: artifacts/training-sessions; auto-builds if present)
  --val-frac FLOAT      Validation fraction for dataset build (default: 0.2)
  --max-per-class N     Optional cap per class for dataset build (default: 0)
  --seed N              Random seed for dataset build (default: 42)
  --skip-train          Skip train_cnn1d.py and reuse existing --model checkpoint
  -h, --help            Show this help and exit
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  print_usage
  exit 0
fi

CONFIG_FILE="${1:-capture_config.json}"
MODEL_PATH="${2:-artifacts/models/cnn_simple/best_model.pt}"
DATA_DIR="${3:-artifacts/processed/cnn_simple}"
OUT_DIR="${4:-artifacts/models/cnn_simple}"
CRITERION="${5:-f1}"
SESSIONS_DIR="${6:-artifacts/training-sessions}"
VAL_FRAC="0.2"
MAX_PER_CLASS="0"
SEED="42"
SKIP_TRAIN=0
SESSIONS_DIR_SET_BY_USER=0

if [[ $# -ge 6 ]]; then
  SESSIONS_DIR_SET_BY_USER=1
fi

if [[ "${1:-}" == --* ]]; then
  CONFIG_FILE="capture_config.json"
  MODEL_PATH="artifacts/models/cnn_simple/best_model.pt"
  DATA_DIR="artifacts/processed/cnn_simple"
  OUT_DIR="artifacts/models/cnn_simple"
  CRITERION="f1"
  SESSIONS_DIR="artifacts/training-sessions"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --config)
        CONFIG_FILE="$2"
        shift 2
        ;;
      --model)
        MODEL_PATH="$2"
        shift 2
        ;;
      --data-dir)
        DATA_DIR="$2"
        shift 2
        ;;
      --output-dir)
        OUT_DIR="$2"
        shift 2
        ;;
      --criterion)
        CRITERION="$2"
        shift 2
        ;;
      --training-sessions-dir)
        SESSIONS_DIR="$2"
        SESSIONS_DIR_SET_BY_USER=1
        shift 2
        ;;
      --val-frac)
        VAL_FRAC="$2"
        shift 2
        ;;
      --max-per-class)
        MAX_PER_CLASS="$2"
        shift 2
        ;;
      --seed)
        SEED="$2"
        shift 2
        ;;
      --skip-train)
        SKIP_TRAIN=1
        shift 1
        ;;
      -h|--help)
        print_usage
        exit 0
        ;;
      *)
        echo "Unknown option: $1" >&2
        echo
        print_usage >&2
        exit 1
        ;;
    esac
  done
fi

THRESHOLD_REPORT="${OUT_DIR}/threshold_report.json"
RUNTIME_CONFIG="${OUT_DIR}/infer_runtime.json"

if [[ -n "${SESSIONS_DIR}" ]]; then
  if [[ -d "${SESSIONS_DIR}" ]] && find "${SESSIONS_DIR}" -mindepth 1 -maxdepth 1 -type d -print -quit | grep -q .; then
    echo "[0/2] Building dataset from sessions directory: ${SESSIONS_DIR}"
    python3 build_cnn_dataset.py \
      --training-sessions-dir "${SESSIONS_DIR}" \
      --output-dir "${DATA_DIR}" \
      --val-frac "${VAL_FRAC}" \
      --max-per-class "${MAX_PER_CLASS}" \
      --seed "${SEED}"
  elif [[ "${SESSIONS_DIR_SET_BY_USER}" -eq 1 ]]; then
    echo "Error: --training-sessions-dir was provided but no session folders were found under: ${SESSIONS_DIR}" >&2
    exit 1
  else
    echo "[0/3] Skipping dataset build (no session folders under default: ${SESSIONS_DIR})"
  fi
fi

if [[ "${SKIP_TRAIN}" -eq 1 ]]; then
  if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "Error: --skip-train was set but model file not found: ${MODEL_PATH}" >&2
    exit 1
  fi
  echo "[1/3] Skipping training; using existing model: ${MODEL_PATH}"
else
  echo "[1/3] Training model"
  python3 train_cnn1d.py \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUT_DIR}"
fi

echo "[2/3] Calibrating threshold"
python3 calibrate_threshold.py \
  --model "${MODEL_PATH}" \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUT_DIR}" \
  --criterion "${CRITERION}"

echo "[3/3] Building runtime inference config"
python3 make_infer_runtime_config.py \
  --config "${CONFIG_FILE}" \
  --threshold-report "${THRESHOLD_REPORT}" \
  --model "${MODEL_PATH}" \
  --out "${RUNTIME_CONFIG}"

echo "Done. Runtime config ready: ${RUNTIME_CONFIG}"
echo "Run inference when ready:"
echo "  python3 run_infer_pipeline.py"
