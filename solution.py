# reference solution
# 1. grab every number-like chunk (6+ digits) as a candidate
# 2. describe it: text on the left, text on the right, its shape (digits -> 9)
# 3. logistic regression: NONE / PHONE / ACCOUNT
# trains from scratch every run, nothing about the test keys is hardcoded
#
#   python solution.py public submission.csv

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

CANDIDATE = re.compile(r"\+?\(?\d[\d .\-()]*\d")
MIN_DIGITS = 6


def candidates(text):
    for m in CANDIDATE.finditer(text):
        if sum(c.isdigit() for c in m.group()) >= MIN_DIGITS:
            yield m.start(), m.end()


def shape(chunk):
    # "+91 98765 43210" -> "+99 99999 99999"
    return re.sub(r"[a-zA-Z]", "a", re.sub(r"\d", "9", chunk))


def describe(text, start, end):
    chunk = text[start:end]
    digits = re.sub(r"\D", "", chunk)
    return {
        "left": text[max(0, start - 40):start].lower(),
        "right": text[end:end + 20].lower(),
        "form": shape(chunk),
        "tokens": f"nd{len(digits)} first{digits[0]} plus{int(chunk.startswith('+'))} "
                  f"groups{len(re.split(r'[ .-]', chunk))}",
    }


def build_table(df, with_labels):
    rows = []
    for row in df.itertuples():
        gold = {}
        if with_labels:
            gold = {(s, e): label for s, e, label in json.loads(row.spans)}
        for start, end in candidates(row.text):
            item = describe(row.text, start, end)
            item.update(id=row.id, start=start, end=end, label=gold.get((start, end), "NONE"))
            rows.append(item)
    return pd.DataFrame(rows)


class Featurizer:
    # char n-grams on left/right context and on the shape, plus a few simple tokens
    def __init__(self):
        self.left = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)
        self.right = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)
        self.form = TfidfVectorizer(analyzer="char", ngram_range=(1, 4), min_df=2)
        self.tokens = CountVectorizer(token_pattern=r"\S+", binary=True)

    def fit_transform(self, t):
        return hstack([self.left.fit_transform(t["left"]), self.right.fit_transform(t["right"]),
                       self.form.fit_transform(t["form"]), self.tokens.fit_transform(t["tokens"])]).tocsr()

    def transform(self, t):
        return hstack([self.left.transform(t["left"]), self.right.transform(t["right"]),
                       self.form.transform(t["form"]), self.tokens.transform(t["tokens"])]).tocsr()


if __name__ == "__main__":
    np.random.seed(0)
    public = Path(sys.argv[1])
    out_path = sys.argv[2]

    train = pd.read_csv(public / "train.csv")
    test = pd.read_csv(public / "test.csv")
    train_table = build_table(train, with_labels=True)
    test_table = build_table(test, with_labels=False)

    feats = Featurizer()
    x_train = feats.fit_transform(train_table)
    x_test = feats.transform(test_table)

    model = LogisticRegression(C=4.0, max_iter=3000, random_state=0)
    model.fit(x_train, train_table.label)
    test_table["pred"] = model.predict(x_test)

    keep = test_table[test_table.pred != "NONE"]
    keep[["id", "start", "end", "pred"]].rename(columns={"pred": "label"}).to_csv(out_path, index=False)
    print(f"candidates train={len(train_table)} test={len(test_table)}, kept {len(keep)}")
