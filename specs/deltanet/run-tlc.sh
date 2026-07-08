#!/usr/bin/env bash
# Run TLC with the throughput-optimized parallel GC (per TLC's startup
# recommendation) for the DeltaNet batch-invariance spec configs.
#
# Usage:
#   ./run-tlc.sh ModalP3
#   ./run-tlc.sh OutlierP0
#
# Add new configs by dropping a PipelineDeterminism_<name>.cfg in this
# directory.

set -euo pipefail

CONFIG="${1:-ModalP3}"
SPEC_DIR="$(cd "$(dirname "$0")" && pwd)"
CFG_FILE="${SPEC_DIR}/PipelineDeterminism_${CONFIG}.cfg"
TLA_FILE="${SPEC_DIR}/PipelineDeterminism.tla"

# Resolve tla2tools.jar in this order:
#   1. $TLA_TOOLS env var (explicit override)
#   2. <repo-root>/.tla-tools/tla2tools.jar (matches CI layout)
#   3. /home/llm/Specula/lib/tla2tools.jar (legacy path on original host)
if [[ -z "${TLA_TOOLS:-}" ]]; then
    REPO_ROOT="$(git -C "${SPEC_DIR}" rev-parse --show-toplevel 2>/dev/null || true)"
    for candidate in \
        "${REPO_ROOT:+${REPO_ROOT}/.tla-tools/tla2tools.jar}" \
        "/home/llm/Specula/lib/tla2tools.jar"; do
        if [[ -n "${candidate}" && -f "${candidate}" ]]; then
            TLA_TOOLS="${candidate}"
            break
        fi
    done
fi

if [[ -z "${TLA_TOOLS:-}" || ! -f "${TLA_TOOLS}" ]]; then
    echo "tla2tools.jar not found. Set \$TLA_TOOLS or place it at <repo>/.tla-tools/tla2tools.jar" >&2
    exit 1
fi

if [[ ! -f "${CFG_FILE}" ]]; then
    echo "Config file not found: ${CFG_FILE}" >&2
    exit 1
fi

cd "${SPEC_DIR}"
exec java -XX:+UseParallelGC -cp "${TLA_TOOLS}" tlc2.TLC \
    -nowarning \
    -config "${CFG_FILE}" \
    "${TLA_FILE}"
