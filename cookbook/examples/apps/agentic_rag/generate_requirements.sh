#!/bin/bash

############################################################################
# Generate requirements.txt from requirements.in using uv with constraints
############################################################################

echo "Generating requirements.txt with constraints"

CURR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

UV_CUSTOM_COMPILE_COMMAND="./generate_requirements.sh" \
  uv pip compile \
    "${CURR_DIR}/requirements.in" \
    --constraint "${CURR_DIR}/constraints.txt" \
    --no-cache \
    --upgrade \
    -o "${CURR_DIR}/requirements.txt"