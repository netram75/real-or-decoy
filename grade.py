# scores a submission. a prediction only counts if id+start+end+label all match exactly.
# final score = average F1 of PHONE and ACCOUNT
#
#   python grade.py --submission submission.csv --by-split

import argparse
import json
import sys

import pandas as pd

LABELS = ("PHONE", "ACCOUNT")


def fail(msg):
    print(json.dumps({"error": msg}))
    sys.exit(1)


def load_submission(path, test):
    try:
        sub = pd.read_csv(path)
    except Exception as e:  # noqa: BLE001
        fail(f"could not read submission: {e}")

    missing = {"id", "start", "end", "label"} - set(sub.columns)
    if missing:
        fail(f"missing columns: {sorted(missing)}")
    if sub.empty:
        return set()

    lengths = dict(zip(test.id, test.text.str.len()))
    unknown = set(sub.id) - set(lengths)
    if unknown:
        fail(f"{len(unknown)} ids not in test.csv, e.g. {sorted(unknown)[:3]}")
    bad_labels = set(sub.label) - set(LABELS)
    if bad_labels:
        fail(f"unknown labels: {sorted(bad_labels)}")
    try:
        sub["start"] = sub.start.astype(int)
        sub["end"] = sub.end.astype(int)
    except ValueError:
        fail("start and end need to be integers")
    bad = sub[(sub.start < 0) | (sub.end <= sub.start) | (sub.end > sub.id.map(lengths))]
    if not bad.empty:
        fail(f"{len(bad)} spans are outside the text, e.g. {bad.iloc[0].to_dict()}")

    return set(zip(sub.id, sub.start, sub.end, sub.label))  # set() also drops duplicates


def load_gold(path):
    gold = set()
    for row in pd.read_csv(path).itertuples():
        for start, end, label in json.loads(row.spans):
            gold.add((row.id, start, end, label))
    return gold


def f1_scores(pred, gold, ids=None):
    if ids is not None:
        pred = {p for p in pred if p[0] in ids}
        gold = {g for g in gold if g[0] in ids}
    out = {}
    for label in LABELS:
        p = {x for x in pred if x[3] == label}
        g = {x for x in gold if x[3] == label}
        tp = len(p & g)
        prec = tp / len(p) if p else 0.0
        rec = tp / len(g) if g else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[label] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}
    out["score"] = round(sum(out[l]["f1"] for l in LABELS) / len(LABELS), 4)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--test", default="public/test.csv")
    ap.add_argument("--labels", default="private/test_labels.csv")
    ap.add_argument("--meta", default="private/test_meta.csv")
    ap.add_argument("--by-split", action="store_true", help="also show normal vs shifted score")
    args = ap.parse_args()

    test = pd.read_csv(args.test)
    pred = load_submission(args.submission, test)
    gold = load_gold(args.labels)
    result = f1_scores(pred, gold)

    if args.by_split:
        meta = pd.read_csv(args.meta)
        result["in_distribution"] = f1_scores(pred, gold, set(meta.id[meta.shifted == 0]))["score"]
        result["shifted"] = f1_scores(pred, gold, set(meta.id[meta.shifted == 1]))["score"]

    print(json.dumps(result, indent=2))
