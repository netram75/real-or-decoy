import json

import pandas as pd
import pytest

from grade import f1_scores, load_gold, load_submission

TEXT = "acct=123456789012, ts=1712345678 call +91 98765 43210"


def span(s, label):
    start = TEXT.index(s)
    return ("a1", start, start + len(s), label)


GOLD = {span("123456789012", "ACCOUNT"), span("+91 98765 43210", "PHONE")}


@pytest.fixture
def test_df():
    return pd.DataFrame({"id": ["a1"], "text": [TEXT]})


def write(tmp_path, rows):
    path = tmp_path / "sub.csv"
    pd.DataFrame(rows, columns=["id", "start", "end", "label"]).to_csv(path, index=False)
    return path


def test_perfect_score():
    assert f1_scores(GOLD, GOLD)["score"] == 1.0


def test_span_has_to_match_exactly():
    # leaving out the +91 is a miss, no partial credit
    pred = {span("123456789012", "ACCOUNT"), span("98765 43210", "PHONE")}
    r = f1_scores(pred, GOLD)
    assert r["PHONE"]["f1"] == 0
    assert r["ACCOUNT"]["f1"] == 1


def test_wrong_label_is_a_miss():
    pred = {span("123456789012", "PHONE"), span("+91 98765 43210", "PHONE")}
    r = f1_scores(pred, GOLD)
    assert r["ACCOUNT"]["f1"] == 0
    assert r["PHONE"]["precision"] == 0.5


def test_decoy_hurts_precision():
    pred = GOLD | {span("1712345678", "PHONE")}
    assert f1_scores(pred, GOLD)["PHONE"]["precision"] == 0.5


def test_duplicates_are_dropped(tmp_path, test_df):
    row = list(span("+91 98765 43210", "PHONE"))
    assert len(load_submission(write(tmp_path, [row, row]), test_df)) == 1


def test_empty_submission_is_ok(tmp_path, test_df):
    assert load_submission(write(tmp_path, []), test_df) == set()


@pytest.mark.parametrize("row", [
    ["zz", 0, 5, "PHONE"],      # id not in test
    ["a1", 0, 5, "EMAIL"],      # unknown label
    ["a1", 5, 5, "PHONE"],      # empty span
    ["a1", 0, 999, "ACCOUNT"],  # runs past the end of the text
])
def test_bad_rows_get_rejected(tmp_path, test_df, row):
    with pytest.raises(SystemExit):
        load_submission(write(tmp_path, [row]), test_df)


def test_missing_column_gets_rejected(tmp_path, test_df):
    path = tmp_path / "sub.csv"
    pd.DataFrame({"id": ["a1"], "start": [0]}).to_csv(path, index=False)
    with pytest.raises(SystemExit):
        load_submission(path, test_df)


def test_load_gold(tmp_path):
    path = tmp_path / "labels.csv"
    spans = [[s, e, label] for _, s, e, label in GOLD]
    pd.DataFrame({"id": ["a1"], "spans": [json.dumps(spans)]}).to_csv(path, index=False)
    assert load_gold(path) == GOLD
