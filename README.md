# Temporal Step Supervision for Robust Step Counting from Wearable Accelerometers

Code for the temporal step supervision method: a U-Net dense predictor
trained with step-timing annotations (expanded into short dense labels)
plus a scheduled hybrid BCE + count-loss objective, for step counting on
wrist/hip/ankle accelerometer data. Evaluated on the Clemson Pedometer
dataset and OxWalk, plus a cross-dataset generalization experiment
between the two.

## Method summary

- **Dense temporal labels**: each annotated step timestamp is expanded
  into a short label interval of width `w` samples, so the model learns
  to localize step evidence in time rather than only predicting an
  aggregate window count.
- **Raster input**: tri-axial acceleration is quantized per-axis into a
  binary time-amplitude map (fold/train-derived percentile clip range,
  16 bins) before being passed to the U-Net.
- **U-Net dense predictor**: encoder/decoder with pooling restricted to
  the amplitude-bin dimension so temporal resolution is preserved
  throughout; outputs a per-sample step probability sequence.
- **Loss**: binary cross-entropy on the dense labels, plus a
  subject-relative count loss (predicted probability mass vs. reference
  count) whose weight ramps in on a schedule (0 during warmup, linear
  ramp to `alpha_max` after) so the model localizes step evidence before
  the aggregate count term is introduced.

## Repository layout

```
step_counting/
├── clemson/
│   ├── code/            model.py, dataset.py, losses.py, train_utils.py, main.py, config.py, metrics.py
│   └── preprocessing/   relabel_motion_energy_025s.py (optional training-label realignment)
├── oxwalk/
│   ├── code/             same module layout as clemson/code
│   └── preprocessing/
└── cross_dataset/
    ├── code/              cross_loaders.py, train_cross_dataset.py (train on one dataset, zero-shot eval on the other)
    └── preprocessing/     resample_oxwalk_25to15hz.py (OxWalk is natively 25Hz; Clemson is 15Hz)
scripts/                   run_clemson.sh, run_oxwalk.sh, run_cross_dataset.sh
analysis/                  aggregate_and_compare.py
```

`clemson/code` and `oxwalk/code` are independent, near-identical module
sets (each has its own `model.py`, `losses.py`, `train_utils.py`, etc.)
because the two datasets differ in sampling rate, file format, fold
protocol, and a few dataset-specific training details (e.g. OxWalk uses
a train/val split with early stopping; Clemson trains for a fixed
120-epoch budget with no validation set, per the leave-two-out protocol
used in prior Clemson step-counting work). `cross_dataset/code` reuses
the dataset-agnostic building blocks from `clemson/code/train_utils.py`
(model, loss, windowing/rasterization) directly.

## Setup

```bash
pip install -r requirements.txt
```

Requires a CUDA GPU (falls back to CPU with `--device cpu`, but training
120 epochs on CPU will be slow).

## Data

Not included in this repository (see each dataset's license/access
terms). Expected layout:

```
data/
├── clemson/                                    Clemson Pedometer dataset
│   └── P001/Regular/*.txt, steps.txt           (one folder per participant x condition)
│       .../SemiRegular/...
│       .../Irregular/...
└── oxwalk/
    ├── OxWalk_Dec2022/Wrist_25Hz/PXX_wrist25.csv        original annotations (25Hz, columns: timestamp,x,y,z,annotation)
    └── OxWalk_Dec2022_peakvalley_025s/Wrist_25Hz/...    motion-energy realigned training labels (see below)
```

- **Clemson Pedometer dataset**: https://cecas.clemson.edu/~ahoover/pedometer/
- **OxWalk**: https://ora.ox.ac.uk/objects/uuid:19d3cb34-e2b3-4177-91b6-1bad0e0163e7

OxWalk's motion-energy realigned training labels (used only as the
*training* target, never for validation/test annotations) are generated
from the original annotations:

```bash
python step_counting/oxwalk/preprocessing/relabel_motion_energy_025s.py \
  --src data/oxwalk/OxWalk_Dec2022 --dst data/oxwalk/OxWalk_Dec2022_peakvalley_025s --radius_sec 0.25
```

Clemson's original annotations are used directly for the main
Clemson experiments; `clemson/preprocessing/relabel_motion_energy_025s.py`
is provided for the same realignment on Clemson if reproducing the
motion-energy-realigned-label ablations.

## Running

```bash
bash scripts/run_clemson.sh    # 4 loss-objective configs x 15 leave-two-out folds
bash scripts/run_oxwalk.sh     # 4 loss-objective configs x 10 subject-wise folds
python analysis/aggregate_and_compare.py
```

Both scripts respect environment variable overrides instead of editing
the script (`PYTHON_BIN`, `CLEMSON_DATA_ROOT`, `OXWALK_REALIGNED_WRIST_ROOT`,
`OXWALK_ORIGINAL_WRIST_ROOT`, `OUTPUT_DIR`) and default to `data/...`
paths relative to the repo root. Each launches 4 configs across 2 GPUs
(`cuda:0`/`cuda:1`); edit the `--device` values inside the script if you
have a different GPU count.

`step_counting/clemson/code/main.py` and `step_counting/oxwalk/code/main.py`
can also be invoked directly with any combination of the CLI flags in
their respective `config.py` (sensor, preprocessing mode, model type,
loss type, label width, etc.) for the input-representation and
preprocessing ablations.

### Cross-dataset generalization

Trains on the full source dataset (all subjects, no held-out fold since
the test set is the other dataset entirely) and evaluates zero-shot on
the full target dataset. Both datasets are unified to 15Hz/wrist so the
raster representation and model architecture are directly comparable.

```bash
python step_counting/cross_dataset/preprocessing/resample_oxwalk_25to15hz.py \
  --src data/oxwalk/OxWalk_Dec2022/Wrist_25Hz --dst data/oxwalk_wrist_15hz
bash scripts/run_cross_dataset.sh
```

## Key hyperparameters (fixed across the main experiments)

| | Clemson | OxWalk |
|---|---|---|
| Sampling rate | 15 Hz | 25 Hz |
| Window length | 60 s (T=900) | 60 s (T=1500) |
| Label width | 5 samples | 7 samples |
| Raster bins | 16 | 16 |
| Optimizer | Adam, lr=3e-4, wd=1e-4, cosine annealing to 1e-5 | same |
| Batch size | 64 | 64 |
| Count-loss `alpha_max` | 0.15 | 0.15 |
| Count-loss warmup / ramp | 30 / 50 epochs (of 120) | 30 / 50 epochs (of 120) |

## Citation

If you use this code, please cite:

```
Zhang, Y., Lago, P., Nguyen, Q.D. Temporal Step Supervision for Robust
Step Counting from Wearable Accelerometers.
```

## Predicting with the final models

`runs/final_{clemson,oxwalk,combined,all3}` hold the final weights. **`all3`** (Clemson + OxWalk + MAREA, `scripts/fit_final_all3.py`) is the one to use if running matters: the paper-trained models undercount running by ~50 %, while held-out cross-validation with MAREA in training gives ~3 % MAPE on running (`scripts/cv_with_marea.py`, `analysis/cv_marea_summary.py`). `step_counting/inference.py`
wraps them so you only supply wrist accelerometer data (x,y,z in g) at any sampling rate;
resampling (15 Hz for clemson/combined/all3, 25 Hz for oxwalk), gravity removal, the saved raster
ranges and 60 s windowing are applied automatically.

```python
from step_counting.inference import StepCounter
res = StepCounter("combined").predict(xyz, fs=30.0)   # xyz: (N,3) array
res.step_count, res.step_times_s, res.prob
```
```bash
python scripts/predict.py recording.csv --model combined --fs 30 --out preds.csv
```
Axis/orientation conventions are not corrected, so wrist placement should resemble the training data.

## Orientation robustness (axes differ between sensors)

The models read raw x/y/z, so a sensor whose axes are mounted or labelled differently is out-of-distribution
(one Mobilise-D right-wrist person: -31 % as-is, ~0 % after the right axis permutation).
`step_counting/orient.py` defines the 24 axis-aligned proper rotations (cube rotation group: identity, 6 quarter-turns,
3 face half-turns, 6 edge half-turns, 8 vertex 120-deg turns; `python step_counting/orient.py` prints them) and their
24 mirror images (left vs right wrist). Eight arms (`step_counting/orient_model.py`): `xyz` (control), `xyz_rot`
(rotation augmentation), `vm` (vector magnitude only), `xyz_vm_drop` (xyz + vm with xyz channel dropout),
`xyz_vm_rot` (xyz + vm + rotation augmentation), and `xyz_mirror` / `xyz_vm_mirror` (the same augmentation drawn from all 48
signed permutations, i.e. rotations plus left/right mirrors; most Mobilise-D wrist sensors are on the left wrist).
`xyz_vm` adds the vm channel with no augmentation, so `xyz` / `xyz_vm` / `xyz_mirror` / `xyz_vm_mirror` form a 2x2
(VM channel x rotation+mirror augmentation). Augmentation is training-only: models are scored as-is, no rotation is chosen at test time.

```bash
python tests/check_orient_parity.py                                   # no-model checks (rasters identical to build_windows)
bash scripts/slurm/submit_gpu.sh scripts/slurm/orient.sbatch          # cv + leave-one-cohort-out + full, all arms (8 array tasks)
python analysis/orient_external.py --device cpu                       # Mobilise-D right wrist under 24 rotations (needs the 'full' models)
python analysis/orient_summary.py                                     # tables + analysis/orient_summary_*.png
python scripts/fetch_mobilised_lab.py --out data/mobilised/lab        # only the Laboratory data.mat files (4 GB, not the 58 GB zips)
python scripts/prepare_mobilised_tvs.py                               # data.mat -> data/mobilised_tvs/*.csv.gz (wrist IMU + INDIP steps)
python analysis/orient_tvs.py --device cuda:0                         # all models as-is on 104 TVS participants, left/right wrist
```
Axis-convention source: Mazza et al., BMJ Open 2021;11:e050785 (`pubs/e050785.full.pdf`), Figure 2 shows the DynaPort MM+
(lower-back) device frame in its gravity-alignment test; it does not define the axes of the INDIP wrist module.

## Paper draft (Quarto + targets)

`paper.qmd` is the paper outline (intro, methods, results, discussion, further analyses). `_targets.R` + `R/functions.R` compute every table,
figure and inline number from the cached results in `analysis/cache`; anything not produced yet shows as "TBD" / a "PENDING" placeholder.
```bash
Rscript -e 'targets::tar_make()'        # builds paper.pdf (Typst); re-run after copying new results into analysis/cache
```
Cluster side: `bash scripts/slurm/run_orient_pipeline.sh` queues the final fits, the TVS scoring, the CV and the leave-one-cohort-out runs.

## Fuller models (place8 / place16 trained on all four cohorts)
`scripts/slurm/orient_fuller.sbatch` trains `xyz_place8` and `xyz_place16` on Clemson + OxWalk + MAREA + Mobilise-D TVS (TVS cropped to its walking bouts) and runs
5-fold participant-level CV over the four cohorts (`scripts/orient_cv.py --cohorts clemson oxwalk marea tvs --tag orient_fuller`). `python analysis/cohort_protocols.py`
writes the cohort comparison (table + `analysis/cohort_protocols.png`). `python tests/check_fuller_records.py` checks the TVS records, folds and bout-restricted scoring without any network.
TVS counts are taken inside INDIP walking bouts padded by 0.5 s (the bout boundaries sit on the first/last step).
