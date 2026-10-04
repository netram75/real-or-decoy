# dumb baseline - only looks at how many digits a number has, ignores the text around it.
# kept here to show why shape alone doesn't work
#
#   python baseline.py public baseline_submission.csv

import re
import sys
from pathlib import Path

import pandas as pd

CANDIDATE = re.compile(r"\+?\(?\d[\d .\-()]*\d")


def label_by_shape(chunk):
    digits = re.sub(r"\D", "", chunk)
    n = len(digits)
    if n == 10 or (n == 12 and digits.startswith("91")) or (n == 11 and digits[0] in "01"):
        return "PHONE"
    if 11 <= n <= 16:
        return "ACCOUNT"
    return None


if __name__ == "__main__":
    public_dir, out_path = sys.argv[1], sys.argv[2]
    test = pd.read_csv(Path(public_dir) / "test.csv")

    rows = []
    for row in test.itertuples():
        for m in CANDIDATE.finditer(row.text):
            label = label_by_shape(m.group())
            if label:
                rows.append({"id": row.id, "start": m.start(), "end": m.end(), "label": label})

    pd.DataFrame(rows, columns=["id", "start", "end", "label"]).to_csv(out_path, index=False)
    print(f"{len(rows)} predictions -> {out_path}")
