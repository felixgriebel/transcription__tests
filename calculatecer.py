import sys
import re
from pathlib import Path

import numpy as np
import pandas as pd

from opencc import OpenCC        
cc = OpenCC("t2s")

# regex: keep digits, letters, spaces and CJK Unified Ideographs
re_keep = re.compile(r"[^\w\s\u4E00-\u9FFF]", flags=re.UNICODE)


def clean(text: str) -> str:
    """Convert to simplified and drop punctuation."""
    if pd.isna(text):
        return ""
    txt = str(text)
    if cc:
        txt = cc.convert(txt)
    return re_keep.sub("", txt).strip()


def cer(ref: str, hyp: str) -> float:
    """Levenshtein distance / len(ref). Simple dynamic-programming version."""
    m, n = len(ref), len(hyp)
    if m == 0:
        return float("nan")

    # distance matrix
    d = np.zeros((m + 1, n + 1), dtype=int)
    d[:, 0] = np.arange(m + 1)
    d[0, :] = np.arange(n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i, j] = min(
                d[i - 1, j] + 1,          # deletion
                d[i, j - 1] + 1,          # insertion
                d[i - 1, j - 1] + cost    # substitution
            )
    return d[m, n] / m


def main(infile: str = "./transcription_files/output_conditioning_false_RAMC.csv", outfile: str = "cer.csv") -> None:
    df = pd.read_csv(infile)

    # figure out which columns hold predicted transcripts
    ref_col = "transcription"
    pred_cols = [c for c in df.columns if c.startswith("transcript_")]

    # clean everything
    for col in [ref_col] + pred_cols:
        df[col] = df[col].apply(clean)

    # compute CER per row/column
    for col in pred_cols:
        cer_col = f"CER_{col}"
        df[cer_col] = [
            cer(r, h) for r, h in zip(df[ref_col], df[col])
        ]

    # show averages
    print("\nAverage CER")
    for col in pred_cols:
        avg = df[f"CER_{col}"].mean()
        print(f"{col}: {avg:.4f}")

    # save file
    df.to_csv(outfile, index=False)
    print(f"\nSaved results to {Path(outfile).resolve()}")


if __name__ == "__main__":
    main()  # accepts 0-2 filenames
