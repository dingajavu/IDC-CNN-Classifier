"""
Compute recall-prioritised operating points for the saved IDC CNN without retraining or the notebook kernel.

What it does
  1. Reloads the saved model and the saved patient split (patient_split.json).
  2. Predicts on the validation and test patches.
  3. For each target sensitivity, picks the HIGHEST threshold on the VALIDATION set that still reaches it,
     then applies that threshold unchanged to the TEST set.
  4. Prints a table, saves operating_points.csv and threshold_analysis.png.
  5. Sanity-checks the pipeline by reproducing the README's 0.5-threshold confusion matrix.

Run inside the idc_cnn environment, from the project folder:
    export IDC_DATASET_DIR=/path/to/your/dataset
    python operating_points.py
Optional: IDC_MODEL_PATH=checkpoints/idc_cnn_best.keras to use a different model file.
"""
import os
import csv
import json
import pathlib

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import keras
import matplotlib
matplotlib.use("Agg")                      # no display needed
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, confusion_matrix

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_DIR = pathlib.Path(os.environ.get("IDC_DATASET_DIR", "/path/to/your/dataset"))
MODEL_PATH = os.environ.get("IDC_MODEL_PATH", "idc_cnn_finalV2.keras")
SPLIT_FILE = pathlib.Path("patient_split.json")
TARGETS = [0.90, 0.95, 0.98]               # target sensitivities to evaluate
PLOT_TARGET = 0.95                         # the one marked on threshold_analysis.png
IMG_SIZE = (50, 50)
CHUNK = 4096                               # patches loaded and predicted at a time

# Values reported in the README at threshold 0.5 (used only as a sanity check)
README_CM = {"TN": 27156, "FP": 4706, "FN": 632, "TP": 6988}

for p, what in [(DATASET_DIR, "dataset folder (set IDC_DATASET_DIR)"),
                (pathlib.Path(MODEL_PATH), "model file (set IDC_MODEL_PATH)"),
                (SPLIT_FILE, "patient split file")]:
    if not p.exists():
        raise FileNotFoundError(f"Missing {what}: {p}")


# ── Custom layers (needed so the saved model can be deserialised) ─────────────
@keras.saving.register_keras_serializable()
class CastToFloat32(keras.layers.Layer):
    def call(self, x):
        return keras.ops.cast(x, "float32")


@keras.saving.register_keras_serializable()
class CastToFloat16(keras.layers.Layer):
    def call(self, x):
        return keras.ops.cast(x, "float16")


model = keras.models.load_model(MODEL_PATH, compile=False)   # compile=False: inference only
print(f"Loaded model: {MODEL_PATH}")


# ── Data helpers (same patient-ID parsing as the notebook) ────────────────────
def extract_patient_id(path):
    """'10253_idx5_x1351_y1101_class0.png' -> '10253_idx5'"""
    parts = os.path.basename(path).split("_")
    pid = ""
    for part in parts:
        if part.startswith("x") and part[1:].isdigit():
            break
        pid += ("_" if pid else "") + part
    return pid


def load_patch(path):
    """Return a 50x50x3 float32 image scaled to [0, 1], as the model expects."""
    img = Image.open(path).convert("RGB")
    if img.size != IMG_SIZE:                  # a few dataset patches are not exactly 50x50
        img = img.resize(IMG_SIZE, Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def predict_paths(paths):
    probs = []
    for i in range(0, len(paths), CHUNK):
        batch = np.stack([load_patch(p) for p in paths[i:i + CHUNK]])
        probs.append(np.asarray(model.predict(batch, batch_size=256, verbose=0)).reshape(-1))
        print(f"  predicted {min(i + CHUNK, len(paths)):,}/{len(paths):,}", end="\r")
    print()
    return np.concatenate(probs)


# ── Rebuild the validation and test sets from the saved split ─────────────────
split = json.load(open(SPLIT_FILE))
val_patients, test_patients = set(split["val"]), set(split["test"])

val_paths, val_labels, test_paths, test_labels = [], [], [], []
for label in (0, 1):
    for p in sorted(DATASET_DIR.glob(f"**/{label}/*.png")):
        pid = extract_patient_id(str(p))
        if pid in val_patients:
            val_paths.append(str(p)); val_labels.append(label)
        elif pid in test_patients:
            test_paths.append(str(p)); test_labels.append(label)
y_val, y_test = np.array(val_labels), np.array(test_labels)
print(f"Validation: {len(y_val):,} patches ({y_val.mean():.1%} positive) | "
      f"Test: {len(y_test):,} patches ({y_test.mean():.1%} positive)")

print("Predicting validation set...")
p_val = predict_paths(val_paths)
print("Predicting test set...")
p_test = predict_paths(test_paths)


# ── Sanity check: reproduce the README numbers at threshold 0.5 ───────────────
def rates(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    return dict(TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp), sens=sens, spec=spec, prec=prec)


print(f"\nTest AUC-ROC: {roc_auc_score(y_test, p_test):.4f} (README: 0.9518) | "
      f"AUC-PR: {average_precision_score(y_test, p_test):.4f} (README: 0.8322)")
r05 = rates(y_test, p_test, 0.5)
print("Confusion matrix at 0.5 vs README:")
for k in ("TP", "FN", "FP", "TN"):
    print(f"  {k}: {r05[k]:>6,}   README {README_CM[k]:>6,}   diff {r05[k] - README_CM[k]:+d}")
print("A diff of 0 means this script reproduces your notebook exactly. Differences of a few patches can come from "
      "resizing the odd non-50x50 patch; large differences mean a different model file.")


# ── Operating points: thresholds chosen on VALIDATION, reported on TEST ───────
fpr_v, tpr_v, thr_v = roc_curve(y_val, p_val)
rows = [dict(name="Default 0.5", threshold=0.5, val_sens=rates(y_val, p_val, 0.5)["sens"], **r05)]
chosen = None
for t in TARGETS:
    idx = int(np.where(tpr_v >= t)[0][0])      # highest threshold that reaches the target
    thr = float(thr_v[idx])
    rows.append(dict(name=f"Target {t:.0%} sensitivity", threshold=thr, val_sens=float(tpr_v[idx]),
                     **rates(y_test, p_test, thr)))
    if abs(t - PLOT_TARGET) < 1e-9:
        chosen = thr

print("\nOperating points (threshold chosen on validation, results on test):")
print(f"{'Operating point':<26}{'Thresh':>8}{'Val sens':>10}{'Test sens':>11}{'Test spec':>11}"
      f"{'Precision':>11}{'FN':>8}{'FP':>9}")
for r in rows:
    print(f"{r['name']:<26}{r['threshold']:>8.4f}{r['val_sens']:>10.4f}{r['sens']:>11.4f}{r['spec']:>11.4f}"
          f"{r['prec']:>11.4f}{r['FN']:>8,}{r['FP']:>9,}")

with open("operating_points.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["name", "threshold", "val_sens", "sens", "spec", "prec", "TP", "FN", "FP", "TN"])
    w.writeheader()
    w.writerows({k: r[k] for k in w.fieldnames} for r in rows)
print("\nSaved operating_points.csv")


# ── Threshold analysis plot (test curves, validation-chosen marker) ───────────
ts = np.linspace(0.05, 0.95, 91)
sens_c, spec_c, f1_c = [], [], []
for t in ts:
    r = rates(y_test, p_test, t)
    sens_c.append(r["sens"]); spec_c.append(r["spec"])
    f1_c.append(2 * r["prec"] * r["sens"] / (r["prec"] + r["sens"]) if r["prec"] + r["sens"] > 0 else 0)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(ts, sens_c, color="#F44336", lw=2, label="Sensitivity (Recall)")
ax.plot(ts, spec_c, color="#2196F3", lw=2, label="Specificity")
ax.plot(ts, f1_c, color="#4CAF50", lw=2, label="F1 Score")
ax.axvline(0.5, color="gray", ls=":", lw=1.5, label="Default: 0.5")
if chosen is not None:
    ax.axvline(chosen, color="purple", ls="--", lw=1.5, label=f"Target {PLOT_TARGET:.0%} sensitivity: {chosen:.3f}")
ax.set_xlabel("Decision Threshold"); ax.set_ylabel("Metric Value")
ax.set_title("Threshold Sensitivity Analysis", fontweight="bold")
ax.grid(True, alpha=0.3); ax.legend()
plt.tight_layout(); plt.savefig("threshold_analysis.png", dpi=150, bbox_inches="tight")
print("Saved threshold_analysis.png")