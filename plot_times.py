# import csv
# from pathlib import Path

# import numpy as np
# import matplotlib.pyplot as plt


# # ── Configuration ────────────────────────────────────────────────
# CSV_PATH = Path("./transcription_files/output_timer2.csv")   # Path to the CSV file
# COLUMN_NUMBER = 6                  # 1 → transcript_1, 2 → transcript_2, 3 → transcript_3, …
# # ─────────────────────────────────────────────────────────────────

# col_name = f"transcript_{COLUMN_NUMBER}"

# # Gather (x, y) pairs
# x_vals, y_vals = [], []
# with CSV_PATH.open(newline="", encoding="utf-8") as f:
#     reader = csv.DictReader(f)
#     if col_name not in reader.fieldnames:
#         raise ValueError(f"Column {col_name} not found in the CSV header.")

#     for row in reader:
#         transcription = row["transcription"]
#         try:
#             time_val = float(row[col_name])
#         except (KeyError, ValueError):
#             continue  # skip rows with missing/non-numeric data

#         x_vals.append(len(transcription))      # transcription length
#         y_vals.append(time_val)                # chosen timing value

# # Fit a straight line:  y = slope * x + intercept
# slope, intercept = np.polyfit(x_vals, y_vals, 1)
# x_fit = np.array([min(x_vals), max(x_vals)])
# y_fit = slope * x_fit + intercept

# # Plot
# plt.figure(figsize=(10, 6))
# plt.scatter(x_vals, y_vals, alpha=0.6, label="Individual samples")
# plt.plot(x_fit, y_fit, color="black", linewidth=2.5, label="Best-fit line")

# # Show slope in the upper-left corner (axes coordinates)
# plt.text(
#     0.03,
#     0.95,
#     f"Slope (Steigung): {slope:.4f} s/char",
#     transform=plt.gca().transAxes,
#     ha="left",
#     va="top",
#     fontsize=10,
#     bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
# )

# plt.title(f"Transcription length vs. processing time\n({CSV_PATH.name}, using {col_name})")
# plt.xlabel("Length of original transcription (characters)")
# plt.ylabel("Time (seconds)")
# plt.grid(True, linestyle="--", alpha=0.5)
# plt.legend()
# plt.tight_layout()
# plt.show()










import csv
import re
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ── Configuration ────────────────────────────────────────────────
CSV_PATH = Path("./transcription_files/output_timer2.csv")      # Path to the CSV file
# ─────────────────────────────────────────────────────────────────
transcript_cols = []             # e.g. ["transcript_1", "transcript_2", …]
data = {}                        # col → list[(length, time)]

with CSV_PATH.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    # pick every column called transcript_<number>
    transcript_cols = [c for c in reader.fieldnames if re.fullmatch(r"transcript_\d+", c)]
    for col in transcript_cols:
        data[col] = []

    for row in reader:
        length = len(row["transcription"])
        for col in transcript_cols:
            try:
                data[col].append((length, float(row[col])))
            except (KeyError, ValueError):
                pass  # skip missing / bad numbers

if not transcript_cols:
    raise RuntimeError("No transcript_x columns found in the CSV.")

# --- 2. Global y-range: start at 0, end at (overall max + 5 %) ---------------
global_max = max(t for pairs in data.values() for _, t in pairs)
pad        = global_max * 0.05           # 5 % head-room
y_limits   = (0, global_max + pad)

# --- 3. Plot sub-plots -------------------------------------------------------
n      = len(transcript_cols)
ncols  = 2 if n > 1 else 1
nrows  = math.ceil(n / ncols)

fig, axes = plt.subplots(
    nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False
)
axes = axes.flatten()

for idx, col in enumerate(transcript_cols):
    ax     = axes[idx]
    pairs  = data[col]
    if not pairs:
        ax.set_visible(False)
        continue

    x_vals, y_vals = zip(*pairs)

    # straight best-fit line
    slope, intercept = np.polyfit(x_vals, y_vals, 1)
    x_fit = np.array([min(x_vals), max(x_vals)])
    y_fit = slope * x_fit + intercept

    ax.scatter(x_vals, y_vals, alpha=0.6)
    ax.plot(x_fit, y_fit, color="black", linewidth=2.0)
    ax.set_title(col)
    ax.set_xlabel("Length (characters)")
    ax.set_ylabel("Time (s)")
    ax.set_ylim(*y_limits)            # <-- common non-negative scale
    ax.grid(True, linestyle="--", alpha=0.5)

    # slope annotation
    ax.text(
        0.03, 0.95,
        f"Slope: {slope:.4f} s/char",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )

# hide unused panels if grid isn’t completely filled
for j in range(idx + 1, nrows * ncols):
    axes[j].set_visible(False)

fig.suptitle(f"Transcription length vs. processing time\n({CSV_PATH.name})", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()