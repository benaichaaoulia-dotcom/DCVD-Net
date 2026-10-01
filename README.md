# DCVD-Net

Research artifact for **Dataset-Conditioned Vessel Decoding (DCVD-Net)** for retinal blood-vessel segmentation from fundus photographs. The core model implementation is intentionally omitted from this public snapshot; commands below that depend on it are not runnable from this version. The implementation and accompanying runnable method code will be included upon acceptance of the paper. Benchmark images, checkpoints, logs, and private environment state are not included.

## Method at a glance

DCVD-Net uses a single-channel U-Net trunk with base width $c=10$ (759,491 parameters) and adds source-indexed affine modulation at decoder levels L3/L2/L1 (40/20/10 channels). For each of three source datasets, the method learns one scale and bias vector per modulated level: 420 additional parameters, 759,911 total. The vectors start at the identity. Each sample is conditioned using its known source-dataset ID.

The trained arm is compared with a parameter-matched twin whose 420 modulation parameters are frozen at identity. Both arms use the same joint training protocol and equal-weight BCE + soft-Dice objective. Training selects checkpoints using the six-image development carve-out; training code never constructs a test loader. Test evaluation is a separate, explicit command requiring `--allow-test` and writing a test-use record.

## Recorded result snapshot

These are the locked measurements recorded for seed 42, threshold 0.6, on the official DRIVE, STARE, and CHASE_DB1 test splits. They describe the experiment; they are not a promise of performance on other data or training environments.

| Measurement | DCVD-Net | Pre-declared bar | Outcome |
|---|---:|---:|---|
| DRIVE thin-sensitivity mean / worst | 0.6937 / 0.5628 | ≥ 0.68 / ≥ 0.55 | Primary axis: pass |
| Maximum sensitivity spread across datasets | 0.0345 | ≤ 0.09 | Secondary: pass |
| DRIVE–CHASE_DB1 macro-DSC gap | 0.0156 | ≤ 0.01 | Secondary: not met |
| Worst-image DSC / maximum count below 0.75 | 0.7378 / 1 per dataset | ≥ 0.69 / ≤ 1 | Secondary: pass |
| Minimum per-dataset macro sensitivity | 0.7845 | ≥ 0.77 | Secondary: pass |
| Parameters | 759,911 | ≤ 1,000,000 | Pass |

The R4 decision rule makes thin-vessel sensitivity decisive; the four secondary axes remain reported, not hidden. F1 passed on two of three evaluated seeds; seed 43 missed both F1 sub-bars. The DSC-gap bar was not met. The data, training, and metric protocol is recorded below.

## Requirements and setup

- Python 3.12
- `uv`
- Linux x86_64 and CUDA 13.0-capable PyTorch for training (locked to PyTorch `2.11.0+cu130`; the experiment used GPU 1 on an RTX 4090 host)
- The three datasets obtained separately under their respective terms; data are not redistributed here

From this repository root:

```sh
uv sync --extra dev
```

`uv.lock` records the resolved Python dependency set. Commands below use `uv run`.

## Dataset layout

Set `data_root` in the YAML config (default `data/retinal`). Expected tree:

```text
data/retinal/
├── DRIVE_official/
│   ├── training/{images,1st_manual}/
│   └── test/{images,1st_manual}/
├── STARE/stare/{ppmdata,labels-ah}/
└── CHASE_DB1/CHASEDB1/
```

This layout uses DRIVE's official train/test annotations, STARE's `ah` labels, and CHASE_DB1's `1stHO` labels. The joint training pool is DRIVE `{21..40}_training`, STARE `{im0001, im0002, im0003, im0004, im0044, im0077, im0139, im0162, im0235, im0239}`, and CHASE_DB1 subjects `01..08` (46 IDs total). The six-image development carve-out is DRIVE `39_training` and `40_training`, STARE `im0235` and `im0239`, and CHASE_DB1 `Image_08L` and `Image_08R`; the remaining 40 images are training-active. The locked official test manifest has DRIVE `01_test`–`20_test`, STARE `{im0081, im0082, im0163, im0236, im0240, im0255, im0291, im0319, im0324}`, and CHASE_DB1 subjects `09..14` (41 IDs total). Training loaders construct only the active-train and development datasets.

Preprocessing is green-channel extraction → gamma correction ($\gamma=1.2$) → CLAHE (clip limit 5, grid 32×32) → resize to 512×512 (bilinear image, nearest-neighbor binary mask) → scaling to the unit interval. Train-only paired rotation selects uniformly from 0°, 15°, 30°, 45°, 90°, 100°, and 120° after green/gamma/CLAHE and before resize. Thin sensitivity uses reference pixels removed by one 3×3 binary erosion; an image contributes only when this stratum contains at least 64 pixels. Obtain data from their authoritative distributors, check their licenses/terms, and do not place data in Git.

## Train the method or matched-budget twin

Training needs a CUDA device; GPU index 1 matches the recorded host. On a single-GPU system, explicitly set `gpu: 0` in the config or pass `--gpu 0`.

```sh
uv run dcvd-train --config configs/train_C.yaml
uv run dcvd-train --config configs/train_T.yaml
```

Training creates a best checkpoint and JSONL/summary records under `artifacts/` (ignored by Git). The schedule is convergence-driven: stop after 20 epochs without dev-loss improvement, retain the maximum-dev-DSC checkpoint, and use 300 epochs only as a cost ceiling. To resume, pass the package-generated `last_*.pth` checkpoint explicitly with `--resume /path/to/last_C.pth`; no checkpoint is auto-discovered. Resume validates the arm and training settings while allowing output path, GPU, budget, and epoch-cap changes.

## Reproduce ablations and seed checks

```sh
for arm in A1 A2 A3 A4; do
  uv run dcvd-train --config "configs/${arm}.yaml"
done
uv run dcvd-train --config configs/C_seed43.yaml
uv run dcvd-train --config configs/C_seed44.yaml
uv run dcvd-train --config configs/T_seed43.yaml
uv run dcvd-train --config configs/T_seed44.yaml
uv run dcvd-train --config configs/C_lr05.yaml
uv run dcvd-train --config configs/C_lr2x.yaml
```

A1 removes source-specific conditioning; A2/A3/A4 omit L3/L2/L1 respectively. Ballast parameters preserve the 759,911 total-parameter budget for these attribution controls, and are not used in the forward pass. Seed-43/44 twin runs and the 0.5×/2× learning-rate sensitivity configs are also included.

## Evaluate locked test splits

Evaluation is deliberately separate from training. It is impossible to run the test command accidentally without the explicit acknowledgment flag:

```sh
uv run dcvd-evaluate --checkpoint artifacts/train/C_s42/best_C.pth --arm C --data-root data/retinal --output artifacts/eval/C_s42_test.json --allow-test
```

The command evaluates all three declared test sets, reports per-image and per-dataset metrics plus F1–F5, and appends the evaluation purpose and split names to `artifacts/eval/test_usage_log.json` (the output directory). The use record is written before test images are opened. Do not use test results for fitting, threshold selection, or checkpoint selection.

## Single-image inference

```sh
uv run dcvd-predict --checkpoint /path/to/best_C.pth --arm C --image /path/to/fundus.png --source DRIVE --output artifacts/prediction.png
```

The source ID is required because the model selects the corresponding learned modulation table. The image is preprocessed with the same green-channel, gamma, CLAHE, and resize steps used in evaluation. Add `--device cpu` to run inference without CUDA.

Evaluation and prediction accept a user-supplied RUN-2026-09-27-06 checkpoint when its architecture arm and parameter count match; the loader translates the earlier encoder/decoder key names. No checkpoint is bundled. Resume is intentionally limited to checkpoints written by this package because deterministic continuation also requires its optimizer and RNG-state metadata.

## Tests and code quality

```sh
uv run pytest
uv run ruff check .
```

The test suite describes intended checks, but it cannot run from this snapshot because the core model implementation is withheld. Dataset-backed training and evaluation also require separately obtained data.

## Scope, provenance, and release status

The core DCVD-Net method implementation is intentionally withheld from this public snapshot and will be released, together with the runnable method code, upon acceptance of the paper.

This snapshot contains supporting files but not a runnable implementation of the method. It does not import or vendor code from the upstream RVS repository. The experiment identified upstream RVS at commit `e1ec21a3f6204e2e6018aeb938852b9cb3f7da79`, but no license file was found there. No license for this repository has been selected. Resolve project ownership, dataset redistribution terms, and licensing before making a public GitHub repository; no dataset, checkpoint, experiment log, credential, or local environment state is included here.
