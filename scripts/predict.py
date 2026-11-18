"""Predict steps from a wrist accelerometer file with one of the final models.

  python scripts/predict.py recording.csv --model combined --fs 30
  python scripts/predict.py recording.csv --model oxwalk --units m/s2 --out preds.csv

Input: CSV with x,y,z columns (or first 3 numeric columns); fs from --fs or a
'timestamp' column in seconds. Resampling to the model's rate and the matching
preprocessing are done automatically (see step_counting/inference.py).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from step_counting.inference import MODEL_SPECS, StepCounter  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("--model", choices=list(MODEL_SPECS), default="combined")
    p.add_argument("--fs", type=float, default=None, help="input sampling rate in Hz")
    p.add_argument("--units", choices=["g", "m/s2"], default="g")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--out", default=None, help="write per-sample probabilities (CSV)")
    a = p.parse_args()

    sc = StepCounter(a.model, device=a.device, threshold=a.threshold)
    r = sc.predict_csv(a.input, fs=a.fs, units=a.units)
    print(f"model={r.model} ({r.model_fs} Hz) duration={r.duration_s:.1f}s")
    print(f"steps (mass)   = {r.step_count:.1f}   ({r.steps_per_minute:.1f}/min)")
    print(f"steps (switch) = {r.step_count_switch}")
    print("per-60s-window:", [round(float(c), 1) for c in r.window_counts])
    if a.out:
        r.to_frame().to_csv(a.out, index=False)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
