# sanity checks on the generated logs and the candidate finder

from generate_data import make_pool
from solution import candidates, key_before


def test_every_real_value_is_a_candidate():
    # if the regex misses a real value, no model can ever get it
    for split in ("train", "test"):
        for row in make_pool(300, split, seed=1):
            found = set(candidates(row["text"]))
            for start, end, _ in row["spans"]:
                assert (start, end) in found, row["text"]


def test_spans_start_and_end_on_the_number():
    for row in make_pool(200, "test", seed=2):
        for start, end, _ in row["spans"]:
            chunk = row["text"][start:end]
            assert chunk[0] in "+(0123456789"
            assert chunk[-1].isdigit()


def test_same_seed_same_data():
    assert make_pool(50, "train", seed=3) == make_pool(50, "train", seed=3)


def test_key_before():
    assert key_before('"payee_acc": "') == "payee_acc"
    assert key_before("ts=1712345678, acct=") == "acct"
    assert key_before("deposit into ") == "into"
