#!/usr/bin/env bash
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/../lib/_common.sh"
source "$SCRIPT_DIR/_shape_template.sh"
matrix_shape_test 8 3
