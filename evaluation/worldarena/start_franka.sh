#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export WORLDARENA_PLATFORM=franka
export WSA_ROBOT_TYPE="${WSA_ROBOT_TYPE:-wr_franka}"
export WSA_ACTION_MODE="${WSA_ACTION_MODE:-}"
export WSA_NORMALIZATION_MODE="${WSA_NORMALIZATION_MODE:-mean_std}"
export WSA_EXECUTE_CHUNK_SIZE="${WSA_EXECUTE_CHUNK_SIZE:-50}"

exec bash "${SCRIPT_DIR}/serve.sh"
