# Encoding Is the Bottleneck — experiment code

Code, frozen index sets and per-cell results for

> **Encoding Is the Bottleneck: Invariance-Induced Blind Spots When Visual Anomaly Detectors Meet Robot Telemetry**
> Chenxuan Zhou, Christy Jie Liang. ACCV 2026 Workshop on Learning from Small Data in Computer Vision (LFSD-CV).

Multivariate robot telemetry from [voraus-AD](https://github.com/vorausrobotik/voraus-ad-dataset)
is rendered as images by eleven encodings (E1–E11) and scored by six training-free visual
anomaly detectors over frozen backbones. Synthetic faults injected into held-out normal windows
supply exact ground truth; byte-identity of the rendered images certifies which faults an
encoding erases before any detector sees them.

```bibtex
@inproceedings{zhou2026encoding,
  title     = {Encoding Is the Bottleneck: Invariance-Induced Blind Spots When Visual
               Anomaly Detectors Meet Robot Telemetry},
  author    = {Zhou, Chenxuan and Liang, Christy Jie},
  booktitle = {Asian Conference on Computer Vision (ACCV) Workshops},
  year      = {2026}
}
```

## Contents

```
voraus-vision/
  voraus_vision/      package: encodings E1-E11, fault injection, detectors, backbones,
                      CLIP patch memory, frozen encoder context
  scripts/NN_*.py     pipeline steps (table below)
  configs/base.yaml   data and encoding constants
results/              frozen index sets and per-cell results (see "Released results")
ref/voraus-ad-dataset upstream voraus-AD code (MVT-Flow, data loader), git submodule
run_pipeline.sh       driver: `reanalysis` (CPU, no dataset) or `full`
requirements.txt      exact package versions used
requirements-reanalysis.txt   the subset needed for `reanalysis` (no PyTorch)
```

Every script resolves paths from its own location (`ROOT = <repo>`), so keep this layout:
`voraus-vision/`, `results/`, `data/` and `ref/` must stay siblings at the repository root.

Comments that cite "plan sec. N" refer to the internal analysis plan written before the main
runs; it is not published. Columns named `prereg_*` (e.g. in `results/21_verdicts/verdicts.csv`)
hold the blindness rule of that plan, which the paper reports as the "planned rule".

## Setup

Python 3.12 is required (the pinned NumPy and SciPy do not support older versions).

```bash
git clone --recurse-submodules https://github.com/chenxuanzhou/ACCV-LFSD.git
cd ACCV-LFSD
python3.12 -m venv venv && source venv/bin/activate    # Windows: py -3.12 -m venv venv; venv\Scripts\activate
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

- A GitHub ZIP download does not contain the submodule; use `git clone`. If you cloned without
  `--recurse-submodules`, run `git submodule update --init`. The submodule is pinned to upstream
  commit `a91a86a`. Do not install the upstream `requirements.txt`; it pins an older stack
  (torch 1.12, pandas 1.5) than the one used here.
- The CUDA 12.6 wheels need a recent NVIDIA driver; for newer GPUs pick the matching CUDA
  index on pytorch.org. On macOS or without a GPU, install torch from the default PyPI index
  (`pip install torch==2.14.0 torchvision==0.29.0`); only `reanalysis` is practical there.
- For `reanalysis` alone, `pip install -r requirements-reanalysis.txt` is enough (no PyTorch).

The results were produced with Python 3.12.10 on Windows 11 and one 8 GB RTX 4060 Laptop GPU.

**Dataset** (needed only for `full`): download
`https://media.vorausrobotik.com/voraus-ad-dataset-100hz.parquet` into `data/`.

```
data/voraus-ad-dataset-100hz.parquet   1,115,942,833 bytes
sha256 c90ab1c78af52651b954d41787f7e89d750f0a128b57600b0e5ceec22621f704
```

**Pretrained weights** (about 1.2 GB) are downloaded from the Hugging Face Hub on first use:
timm `wide_resnet50_2` (resolves to `wide_resnet50_2.racm_in1k`), timm
`vit_base_patch14_dinov2.lvd142m`, and open_clip `ViT-B-16-quickgelu` / `openai`.

## Reproducing

```bash
./run_pipeline.sh reanalysis   # CPU only, no dataset
./run_pipeline.sh full         # everything from the dataset (CUDA GPU, ~17 GB of disk)
```

`reanalysis` recomputes the main grid from the released per-image scores (per-cell AUROC,
bootstrap CIs, `07`) and everything built on it: verdicts (`21`), ANOVA (`11`, `30`,
`_anova_pools`), the fixed-detector few-shot curve (`30`), the summary tables (`13`) and the
checklist (`16`). Afterwards `git diff --stat` should be empty; on Windows git may warn that
CRLF will be replaced by LF in the rewritten files, which is only the line ending. The other
result files are reproduced only by `full`.

Set `PYTHON=...` to choose the interpreter. Notes for `full`:

- **Steps that skip or resume.** 05, 06, 12 and 15 skip outputs that already exist; `09` skips
  R2 because `r2_scores.parquet` is released; `29 image`, `29 mvtflow`, `29 mvtflow_iters` and
  `32` resume from their released CSVs. As released, `full` therefore re-renders nothing that is
  needed for scoring and re-scores nothing. To recompute, delete first:
  `results/06_scores/` (steps 06, 12, 15; this also makes 06 write the anomaly maps that `10`
  needs), `results/09_signal/r2_scores.parquet`,
  `results/29_intermediate_k/{image_intermediate_k,mvtflow_intermediate_k,mvtflow_const_iters}.csv`,
  `results/32_kshot_statistics/kshot_statistics.csv`.
- **Do not delete** `results/05_render/manifest.csv`, `manifest_ext.csv` or `injection_gt*.csv`.
  They are frozen inputs that fix which window receives which fault. `05` rebuilds a missing
  manifest from the current fault list, which now also contains the controls added by `12`, so a
  rebuilt manifest has 9,722 rows instead of 8,962 and no longer matches the released scores.
- **Frozen index sets.** Steps 01 and `29 indices` rewrite the four index sets. With the pinned
  NumPy they reproduce the released files exactly; the driver checks this with `git diff` and
  stops if they differ. Steps 00, 01 and 02 also rewrite `results/00_precheck/*`, `meta.csv`,
  `norm_stats.json` and `results/02_context/*`; these are not checked automatically (the PCA
  basis in `context.npz` can flip sign under another BLAS), so compare them with `git diff`.
- `08_diagnose_controls.py` must be run three times (as in the driver) to rebuild
  `results/08_diagnose/control_sweep_all.csv` in the released row order.
- `09` trains R1 (MVT-Flow at L = 512, not used in the paper), and `13` then adds R1 rows to
  `t5_signal_domain.csv` and `t5b_signal_domain_synthetic.csv`.
- `10` runs only when anomaly maps exist (see the first note); the driver skips it otherwise.

### Pipeline steps

| Step | What it does | Needs |
|---|---|---|
| `00_precheck` | dataset facts: channels, sampling rate, lengths, official split, category counts | dataset |
| `01_preprocess` | drop constant channel (130→129), resample to L=512, train statistics, frozen holdout and few-shot indices | dataset |
| `01b_cache_raw_holdout` | native-length holdout recordings (duration fault S5 is applied before resampling) | dataset |
| `02_fit_context` | frozen encoder context: PCA-3, scales, E4 channels | 01 |
| `03_verify_invariance` | byte-level invariance probe on 8 windows | 01, 01b |
| `05_render` | renders 8,962 images per encoding | 01, 01b, 02 |
| `06_run_detectors` | detectors M1–M5 over k ∈ {1, 2, 4, 8, 16, all} × 5 seeds | GPU, 05 |
| `07_evaluate` | per-cell AUROC, AUPRC, F1-max and bootstrap CIs | released scores |
| `08_diagnose_controls` | spike/hold amplitude × width × number of affected channels | GPU, 01, 02 |
| `08b_grid_ablation`, `08c_layer_ablation` | patch-grid and backbone-stage ablations | GPU, 01, 02 |
| `09_signal_baselines` | signal-domain PCA reconstruction (R2) and MVT-Flow at L=512 (R1); neither is used in the paper | 01, 01b |
| `10_attribution` | channel attribution and temporal localisation from anomaly maps; not used in the paper | 06 maps |
| `11_variance_decomposition` | two-way ANOVA of encoding and detector | released |
| `12_extend_controls` | footprint-matched controls S7m/S8m (8 channels), S7g/S8g (all) | GPU, 05 |
| `13_final_tables` | summary tables | released |
| `14_channel_permutation` | shuffle the channel rows of the heatmap encodings | GPU, 01, 02 |
| `15_winclip` | M6: patch memory on CLIP ViT-B/16 tokens in the style of WinCLIP+ | GPU, 05 |
| `16_checklist` | encoding checklist from the verdicts | released |
| `17_affine_content` | affine-explained share of real anomalies; correlation with AUROC gaps | 01, 07 |
| `18_detour_decomposition` | rendering steps applied in the signal domain | 01 |
| `19_length_information` | MVT-Flow with padding vs resampling | dataset |
| `20_r1_gap` | MVT-Flow through the official pipeline, 948 vs 758 recordings | dataset |
| `21_verdicts` | proved blind / partial / visible per (encoding, fault) | released |
| `22_mvtflow_fewshot` | MVT-Flow at k ∈ {2, 4, 8, 16, all} with the published 70-epoch schedule (k = 1 cannot be fitted) | dataset |
| `23_verify_on_benchmark` | byte identity over all 190 windows × 396 cells | 05 |
| `24_matched_channel_signal` | signal-domain kNN / PCA on the same 28 or 129 channels | 01 |
| `29_intermediate_k` | k ∈ {32, 64, 128, 256}. Stages: `indices` (01), `signal` (01), `image` (GPU, 05), `mvtflow` (GPU, dataset), `mvtflow_iters` (GPU, dataset; k = 2–256 at a fixed 1680 optimisation steps) | see stages |
| `30_revision_reanalysis` | fixed-detector few-shot curve, ANOVA with interaction | released |
| `31_density_controls` | PaDiM on raw signal bins and on raw E11 pixels | GPU, 01, 05 |
| `32_kshot_statistics` | normalisation statistics estimated from the k bank windows only | GPU, 01 |
| `_anova_pools` | encoding/detector ratio without E11, on E1–E8, without the CLIP detector | released |

"released" means the step runs from the files in this repository alone. The `*_ext.parquet`
score files (controls of `12`) exist for M1–M5 only.

## Where each result in the paper comes from

Numbers in the paper are rounded; the files hold full precision. Sections refer to the
camera-ready version.

| Paper | Results file(s) | Script |
|---|---|---|
| Sec. 3 / 5.1: the 19 blind (encoding, fault) pairs, 82 (pair, severity) cells, 77 byte-identical and 5 up to rounding; 396 cells checked | `results/23_benchmark_verify/byte_identity.csv`, `results/21_verdicts/verdicts.csv` | 23, 21 |
| Sec. 3: rounding residual (values that differ, AUROC near 0.5) | `results/23_benchmark_verify/byte_identity.csv`, `results/07_eval/pb_sweep.csv` | 23, 07 |
| Sec. 3: byte-level probe (refuted invariances E3 under gain, E8 under offset; unpredicted E6 under offset; STFT grey levels under shift; duration residual) | `results/03_invariance/invariance_check.csv`, `invariance_check_scored.csv` | 03 |
| Sec. 3: E4 renders 9 of 129 channels; single-channel faults missing the rendered set | `results/02_context/context_summary.json`; column `injected_ch_rendered_by_e4` of `results/05_render/manifest.csv` | 02, 05 |
| Sec. 4: data, split, holdout, index sets, fault suite | `results/00_precheck/precheck.json`, `results/01_preprocess/*.json`, `results/05_render/manifest*.csv`, `injection_gt*.csv` | 00, 01, 05, 12 |
| Sec. 4 deviations: null s.e. of the planned blindness rule | `results/21_verdicts/power.json` | 21 |
| Fig. 1c: PatchCore scores, k = 4 | `results/06_scores/<E>/M1_patchcore_wrn50.parquet` | 06 |
| Fig. 1d: AUROC per (encoding, fault); dark cells; triangles | `results/07_eval/pb_sweep.csv`; `byte_identity.csv` / `verdicts.csv`; `injected_ch_rendered_by_e4` in `manifest.csv` | 07, 23, 21 |
| Sec. 5.2: encodings span 0.587–0.909, detector spread, heatmap family | `results/07_eval/pa_percategory.csv` (also the `k = all` row of `results/13_tables/t2_fewshot_curve.csv`) | 07, 13 |
| Sec. 5.2: ANOVA with interaction; ratio sensitivity to the pools | `results/30_reanalysis/anova_with_interaction.csv`; printed by `_anova_pools.py` | 30, `_anova_pools` |
| Sec. 5.2: same channels in the signal domain | `results/24_matched_channel/matched_channel.csv`, `matched_channel_best.csv` | 24 |
| Sec. 5.2: channel-row shuffle | `results/14_permutation/permutation_summary.csv` | 14 |
| Sec. 5.2: affine share of real anomalies, r = 0.67 / 0.60 | `results/17_affine/affine_vs_gap.csv`, `affine_by_category.csv` | 17 |
| Fig. 2a, Sec. 5.3: amplitude vs number of affected channels | `results/08_diagnose/control_sweep_all.csv` | 08 |
| Sec. 5.3: single-, 8- and all-channel controls | `results/07_eval/pb_sweep.csv` (faults S7, S8, S7m, S8m, S7g, S8g) | 07, 12 |
| Sec. 6: finer patch grid | `results/08_diagnose/grid_ablation.csv` | 08b |
| Sec. 6: share of variance kept by the projection (47.4%) | `pca_evr_sum` in `results/02_context/context_summary.json` | 02 |
| Table 1 / Fig. 2b: MVT-Flow, 70 epochs | `results/22_mvtflow_fewshot/mvtflow_fewshot.csv`; k ≥ 32 `results/29_intermediate_k/mvtflow_intermediate_k.csv` | 22, 29 |
| Table 1: MVT-Flow, 1680 steps | `results/29_intermediate_k/mvtflow_const_iters.csv`, k = 2–256 (at k = all both schedules coincide: `mvtflow_fewshot.csv`) | 29, 22 |
| Table 1: signal kNN (28 channels, z-score) | `results/24_matched_channel/matched_channel.csv`; k ≥ 32 `results/29_intermediate_k/signal_28ch_zscore.csv` | 24, 29 |
| Table 1: E11 pixels + PaDiM; Sec. 5.4: the same Gaussian on the signal (0.707) | `results/31_density_controls/density_controls.csv` | 31 |
| Table 1: E11 + PatchCore, E11 + PaDiM | `results/30_reanalysis/fewshot_dispersion_fixed_detector.csv` (rule `fixed`); k ≥ 32 `results/29_intermediate_k/image_intermediate_k.csv` | 30, 29 |
| Sec. 5.4: MVT-Flow with 948 vs 758 recordings | `results/20_r1_gap/r1_gap.csv` | 20 |
| Sec. 5.4: k-shot normalisation statistics | `results/32_kshot_statistics/kshot_statistics.csv` | 32 |

Not computed by a script:

- The 95% intervals for r = 0.67 / 0.60 (Sec. 5.2) are Fisher-z intervals over the 12 anomaly
  categories, tanh(atanh(r) ± 1.96/√(12 − 3)), with r from column `pearson_r_for_pair` of
  `affine_vs_gap.csv`.
- The rings in Fig. 1d mark the cells where the internal plan's prediction was wrong:
  predicted blind but visible (E3 × S2, E8 × S1, E7 × S4b, E8 × S4b) and predicted visible but
  blind (E6 × S1). The observed outcomes are in `invariance_check.csv`.
- Panels 1a/1b of Fig. 1 show renderings and need the dataset.

## Analyses superseded during revision

Some files hold results of an earlier draft or internal report that the paper no longer uses or
reports differently. They are kept because the code produces them; do not quote them as results
of the paper.

| File | Earlier use | What the paper uses instead |
|---|---|---|
| `results/11_anova/anova_by_k.csv` | headline "encoding explains 20.5× the detector variance"; its pooled row also mixes in zero-shot rows (k = 0) | the same ratio reported with the interaction term and its pool sensitivity (`30`, `_anova_pools`) |
| `results/22_mvtflow_fewshot/mvtflow_fewshot.csv` at k ≤ 16 | read as "MVT-Flow is below chance with few samples" | the numbers stay in Table 1 ("70 epochs"), but that schedule gives only 70 gradient steps at small k; the matched-budget comparison is `29 mvtflow_iters` ("1680 steps") |
| `results/13_tables/t2_fewshot_curve.csv`, rows k < all | few-shot curve with the best of six detectors per k | fixed detectors (`results/30_reanalysis/fewshot_dispersion_fixed_detector.csv`) |
| `results/07_eval/pb_verdict.csv`, AUROC-based columns of `results/21_verdicts/verdicts.csv` | blindness judged by a bootstrap rule | byte identity only: column `byte_identical` of `verdicts.csv` (from the 8-window probe of `03`), verified over all 190 windows by `23` |
| `results/19_length/length_arms.csv`, `results/18_detour/*` | duration cost 0.046 AUROC; detour ladder | not reported |
| rows with `k = "0"` in `results/06_scores/*/M6_winclip.parquet` | zero-shot WinCLIP | not reported; M6 is used as a few-shot patch memory |
| `results/09_signal/r2_scores.parquet` | signal-domain PCA reconstruction (R2) as the reference | kNN and PCA-20 on the same channels (`24`) and MVT-Flow via the official pipeline (`20`, `22`, `29`) |

## Released results

`results/` contains the frozen inputs every later step reads and the per-cell outputs:

- index sets: `01_preprocess/fewshot_indices.json`, `holdout_normal.json`, `kept_channels.json`,
  `29_intermediate_k/fewshot_indices_ext.json`
- fault ground truth: `05_render/manifest.csv`, `manifest_ext.csv`, `injection_gt.csv`, `injection_gt_ext.csv`
- frozen encoder context: `02_context/context.npz`, `context_summary.json`; training statistics `01_preprocess/norm_stats.json`
- per-image anomaly scores for every encoding and detector, on the test split for every k and
  seed and on the synthetic faults at k = 4: `06_scores/<E>/<M>.parquet`, and for the
  controls of `12`, `<M>_ext.parquet` (M1–M5)
- every evaluation, verdict, ablation and summary file written by the steps above

## What is not in this repository

| Not included | Size | Regenerate with |
|---|---|---|
| the voraus-AD parquet | 1.1 GB | download (see Setup) |
| resampled windows `01_preprocess/windows.npy`, recording lengths `raw_lengths.npy`, native-length holdout `holdout_raw.npz` | 0.6 GB | 01, 01b |
| rendered images `05_render/E*/images*.npy` | 15 GB | 05, 12 |
| anomaly maps `06_scores/*/*_maps.npz` | 0.3 GB | 06 (delete the scores first) |
| attribution tables `10_attribution/` | 33 MB | 10, after 06 |
| native-length cache `19_length/native.npz` | 0.8 GB | 19 |
| MVT-Flow weights trained by 20, 22, 29 | — | retrained from scratch by each step |
| `04_mvtflow_repro/`: a run of the upstream `ref/voraus-ad-dataset/train.py` (AUROC 0.922; weights 8 MB and log), not used in the paper | 8 MB | upstream `train.py` with `DATASET_PATH` pointing to `data/` |
| scripts that typeset the paper's figures and tables from these files | — | — |

The excluded windows, caches and images are copies or renderings of the dataset; the anomaly
maps are large intermediate outputs.

## Data and licences

The voraus-AD dataset (J. T. Brockmann, M. Rudolph, B. Rosenhahn, B. Wandt, "The voraus-AD
dataset for anomaly detection in robot applications", IEEE Trans. Robot. 40:438–451, 2024,
doi:10.1109/TRO.2023.3332224) is licensed CC BY-NC-SA 4.0 and is not redistributed here. The
result files in `results/` are derived from it and are provided under the same licence,
CC BY-NC-SA 4.0. The upstream code in `ref/voraus-ad-dataset` is MIT-licensed by its authors.
