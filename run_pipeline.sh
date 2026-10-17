#!/usr/bin/env bash
# Pipeline driver.  Run from anywhere; paths are resolved relative to this file.
#
#   ./run_pipeline.sh reanalysis   CPU only, no dataset needed: recompute every
#                                  AUROC table, verdict and summary from the
#                                  released per-image scores and per-cell files.
#   ./run_pipeline.sh full         everything from the raw dataset (needs the
#                                  parquet in data/, one CUDA GPU, ~1.2 GB of
#                                  pretrained weights downloaded on first use).
#
# Set PYTHON to choose the interpreter (default: python).
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
S=voraus-vision/scripts

run() { echo; echo "=== $* ==="; "$PY" "$S/$1" "${@:2}"; }

# Files that are frozen and released.  01_preprocess and 29_intermediate_k
# "indices" rewrite them; with the pinned NumPy they reproduce the released
# versions byte for byte, and this check makes any difference loud.
FROZEN="results/01_preprocess/fewshot_indices.json results/01_preprocess/holdout_normal.json
results/01_preprocess/kept_channels.json results/29_intermediate_k/fewshot_indices_ext.json"
check_frozen() {
  if git diff --quiet -- $FROZEN; then
    echo "[frozen] index sets unchanged"
  else
    echo "[frozen] WARNING: regenerated index sets differ from the released ones:" >&2
    git diff --stat -- $FROZEN >&2
    echo "[frozen] restore them with: git checkout -- $FROZEN" >&2
    exit 1
  fi
}

check_dataset() {
  local f=data/voraus-ad-dataset-100hz.parquet
  if [ ! -f "$f" ]; then
    echo "missing $f -- download https://media.vorausrobotik.com/voraus-ad-dataset-100hz.parquet" >&2
    exit 1
  fi
  "$PY" - "$f" <<'EOF'
import hashlib, sys
h = hashlib.sha256()
with open(sys.argv[1], "rb") as fh:
    for block in iter(lambda: fh.read(1 << 24), b""):
        h.update(block)
want = "c90ab1c78af52651b954d41787f7e89d750f0a128b57600b0e5ceec22621f704"
if h.hexdigest() != want:
    sys.exit(f"sha256 mismatch: {h.hexdigest()} != {want}")
print("[data] sha256 OK")
EOF
}

reanalysis() {
  run 07_evaluate.py                 # per-cell AUROCs and bootstrap CIs from 06_scores
  run 21_verdicts.py                 # proved blind / partial / visible (needs 03 + 07)
  run 11_variance_decomposition.py
  run _anova_pools.py                # pool sensitivity of the encoding/detector ratio
  run 30_revision_reanalysis.py      # fixed-detector few-shot curve, ANOVA with interaction
  run 13_final_tables.py
  run 16_checklist.py
}

full() {
  check_dataset
  run 00_precheck.py
  run 01_preprocess.py;  check_frozen
  run 01b_cache_raw_holdout.py
  run 02_fit_context.py
  run 03_verify_invariance.py
  run 05_render.py                   # ~14 GB of images; skips encodings already rendered
  run 06_run_detectors.py            # GPU; skips (encoding, detector) pairs whose scores exist
  run 12_extend_controls.py          # GPU; footprint-matched controls S7m/S8m/S7g/S8g
  run 15_winclip.py                  # GPU
  run 07_evaluate.py
  run 23_verify_on_benchmark.py      # byte-identity over all 190 windows x 396 cells
  run 21_verdicts.py
  # the released control_sweep_all.csv came from these three invocations, in this order
  run 08_diagnose_controls.py E1
  run 08_diagnose_controls.py E2 E4 E5 E7 E3 E6 E8 E9
  run 08_diagnose_controls.py E10 E11
  run 08b_grid_ablation.py
  run 08c_layer_ablation.py
  run 09_signal_baselines.py
  run 10_attribution.py              # needs the anomaly maps written by 06
  run 11_variance_decomposition.py
  run 14_channel_permutation.py
  run 17_affine_content.py
  run 18_detour_decomposition.py
  run 19_length_information.py
  run 20_r1_gap.py                   # MVT-Flow through the official pipeline (948 vs 758)
  run 22_mvtflow_fewshot.py
  run 24_matched_channel_signal.py
  run 29_intermediate_k.py indices;  check_frozen
  run 29_intermediate_k.py signal
  run 29_intermediate_k.py image
  run 29_intermediate_k.py mvtflow
  run 29_intermediate_k.py mvtflow_iters
  run 30_revision_reanalysis.py
  run 31_density_controls.py
  run 32_kshot_statistics.py
  run _anova_pools.py
  run 13_final_tables.py
  run 16_checklist.py
}

case "${1:-}" in
  reanalysis) reanalysis ;;
  full)       full ;;
  *) echo "usage: $0 reanalysis|full" >&2; exit 2 ;;
esac
echo; echo "=== done ==="
