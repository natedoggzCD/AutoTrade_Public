#!/usr/bin/env bash
set -euo pipefail

mkdir -p "${DOWNDAY_ROOT:-/workspace/data/downday}" /workspace/logs /workspace/reports /workspace/plans

exec "$@"
