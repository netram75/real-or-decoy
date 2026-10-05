# reference solution
# 1. grab every number-like chunk (6+ digits) as a candidate
# 2. describe it: text on the left, text on the right, its shape (digits -> 9),
#    and the key right before it split into parts (payee_acc -> payee acc)
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


def key_before(left):
    # last word-ish thing before the number: "payee_acc=", "\"tel\": \"", "deposit into "
    m = re.search(r"([a-z][a-z_/#.]*)[^a-z]*$", left)
    return m.group(1) if m else ""


def describe(text, start, end):
    chunk = text[start:end]
    digits = re.sub(r"\D", "", chunk)
    left = text[max(0, start - 40):start].lower()
    key = key_before(left)
    return {
        "left": left,
        "right": text[end:end + 20].lower(),
        "form": shape(chunk),
        "tokens": f"nd{len(digits)} first{digits[0]} plus{int(chunk.startswith('+'))} "
                  f"groups{len(re.split(r'[ .-]', chunk))}",
        "key": key,
        "parts": " ".join(p for p in re.split(r"[^a-z]+", key) if p),
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
    # char n-grams on left/right context and on the shape, plus a few simple tokens.
    # the key gets its own n-grams so it isn't drowned out by whatever else is in the left 40 chars
    def __init__(self):
        self.left = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)
        self.right = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)
        self.form = TfidfVectorizer(analyzer="char", ngram_range=(1, 4), min_df=2)
        self.tokens = CountVectorizer(token_pattern=r"\S+", binary=True)
        self.key = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1, sublinear_tf=True)
        self.parts = CountVectorizer(token_pattern=r"\S+", binary=True)
        self.cols = ["left", "right", "form", "tokens", "key", "parts"]

    def fit_transform(self, t):
        return hstack([getattr(self, c).fit_transform(t[c]) for c in self.cols]).tocsr()

    def transform(self, t):
        return hstack([getattr(self, c).transform(t[c]) for c in self.cols]).tocsr()


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
