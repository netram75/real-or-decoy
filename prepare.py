# splits raw/ into public/ (what solvers get) and private/ (hidden answers)
# test = 1000 normal lines held out from train + all shifted lines

import argparse
import json
import random
import uuid
from pathlib import Path

import pandas as pd


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="raw")
    ap.add_argument("--public", default="public")
    ap.add_argument("--private", default="private")
    ap.add_argument("--holdout", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    r = random.Random(args.seed)
    train_pool = read_jsonl(Path(args.raw) / "train_pool.jsonl")
    test_pool = read_jsonl(Path(args.raw) / "test_pool.jsonl")
    r.shuffle(train_pool)

    train_rows = train_pool[args.holdout:]
    test_rows = train_pool[:args.holdout] + test_pool
    r.shuffle(test_rows)  # so nobody can guess shifted vs normal from order

    def new_id():
        return uuid.UUID(int=r.getrandbits(128)).hex[:12]

    pub = Path(args.public)
    priv = Path(args.private)
    pub.mkdir(parents=True, exist_ok=True)
    priv.mkdir(parents=True, exist_ok=True)

    train = pd.DataFrame(
        [{"id": new_id(), "text": row["text"], "spans": json.dumps(row["spans"])} for row in train_rows]
    )
    test_ids = [new_id() for _ in test_rows]
    test = pd.DataFrame({"id": test_ids, "text": [row["text"] for row in test_rows]})
    labels = pd.DataFrame({"id": test_ids, "spans": [json.dumps(row["spans"]) for row in test_rows]})
    meta = pd.DataFrame({"id": test_ids, "shifted": [int(row["split"] == "test") for row in test_rows]})

    train.to_csv(pub / "train.csv", index=False)
    test.to_csv(pub / "test.csv", index=False)
    labels.to_csv(priv / "test_labels.csv", index=False)
    meta.to_csv(priv / "test_meta.csv", index=False)  # only for my own analysis

    # one example row so the format is clear
    pd.DataFrame([{"id": test_ids[0], "start": 0, "end": 1, "label": "PHONE"}]).to_csv(
        pub / "sample_submission.csv", index=False
    )

    n_shift = int(meta.shifted.sum())
    print(f"train {len(train)} | test {len(test)} ({n_shift} shifted, {len(test) - n_shift} normal)")
