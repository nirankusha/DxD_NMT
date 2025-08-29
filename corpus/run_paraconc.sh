# -*- coding: utf-8 -*-
#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_paraconc.sh [OUTDIR] [SCRIPT]
#   OUTDIR  (optional) output directory, default: results
#   SCRIPT  (optional) path to the Python CLI, default: sketh_paraconc.py
#

OUTDIR="${1:-results}"
SCRIPT="${2:-sketch_paraconc.py}"

: "${SKETCHENGINE:?Set SKETCHENGINE environment variable with your API key}"

mkdir -p "$OUTDIR"

python sketch_paraconc.py \
  -q 'q[lemma="kobieta"]' 'q[lemma="kobieta"][lemma="wejść"]' 'q[lemma="wejść"][lemma="kobieta"]' \
  --outdir results --save
"""
Created on Mon Aug 11 16:03:15 2025

@author: niran
"""

