#!/usr/bin/env bash
# Runs the pipeline stages in order. Usage: ./run_all.sh [configs/pipeline.yaml]
set -euo pipefail
CONFIG="${1:-configs/pipeline.yaml}"

python -m pipeline.s01_frames --config "$CONFIG"
python -m pipeline.s02_predict --config "$CONFIG"
python -m pipeline.s03_fuse --config "$CONFIG"
python -m pipeline.s03b_zoom --config "$CONFIG"
python -m pipeline.s04_track --config "$CONFIG"
python -m pipeline.s04b_recheck --config "$CONFIG"
python -m pipeline.s04c_analysis --config "$CONFIG"   # report only
python -m pipeline.s05_triage --config "$CONFIG"
# manual step: review the eval tasks in CVAT, export "CVAT for video 1.1" (+ "for images 1.1")
# into datasets/annotations/cvat_eval/. Without the exports s06 stops the script here.
python -m pipeline.s06_import --config "$CONFIG"
python -m pipeline.s07_dataset --config "$CONFIG"
python -m pipeline.s08_train --config "$CONFIG"
python -m pipeline.s09_evaluate --config "$CONFIG" --runs n_step3 n_step6 n_step9          # val: pick run + threshold (max F1 macro over bands)
python -m pipeline.s09_evaluate --config "$CONFIG" --runs best stages --split eval --video-clips all   # eval, once: chosen run + every stage
