# makes fake server logs with real phone/account numbers + lookalike decoys
# train pool = normal logs, test pool = shifted logs (new keys, new layouts, new decoys)
# everything is random, no real data

import argparse
import json
import random
import string
from pathlib import Path

DIGITS = string.digits

KEYS = {
    "train": {
        "PHONE": ["phone", "mobile", "contact_no", "cust_phone", "whatsapp"],
        "ACCOUNT": ["acct", "account_no", "bank_acct", "beneficiary_acct"],
        "DECOY": ["ts", "order_id", "txn_ref", "tracking_no", "build", "req_id", "invoice_no", "epoch_ms"],
    },
    # only used in the shifted pool
    "test": {
        "PHONE": ["msisdn", "tel", "alt_contact", "caller"],
        "ACCOUNT": ["ac_no", "payee_acc", "savings_acc"],
        "DECOY": ["awb", "booking_ref", "ack_no", "seq", "otp"],
    },
}

PHRASES = {
    "train": {
        "PHONE": ["pls call {v}", "reach me on {v}", "my number is {v}"],
        "ACCOUNT": ["transfer to a/c {v}", "credit account {v}"],
        "DECOY": ["order {v} shipped", "ticket {v} closed", "txn {v} settled"],
    },
    "test": {
        "PHONE": ["ring {v} after 6", "ping me at {v}", "text {v} when done"],
        "ACCOUNT": ["deposit into {v} (savings)", "payout to acc {v}"],
        "DECOY": ["booking {v} confirmed", "awb {v} out for delivery", "ack {v} received"],
    },
}

PHONE_STYLES = {
    "train": ["plain", "plus91_space", "plus91_dash", "zero", "us_paren", "us_dash", "us_plus1"],
    "test": ["plain", "plus91_space", "plus91_334", "dash_91", "spaced_433", "us_dots", "uk"],
}
ACCOUNT_STYLES = {"train": ["plain", "grouped4"], "test": ["plain", "grouped4", "dashed4"]}


def rand_digits(r, n, first=None):
    head = first if first is not None else r.choice(DIGITS)
    return str(head) + "".join(r.choice(DIGITS) for _ in range(n - 1))


def make_phone(r, split):
    style = r.choice(PHONE_STYLES[split])
    d = rand_digits(r, 10, first=r.randint(6, 9))  # indian mobile, starts 6-9
    if style == "plain":
        return d
    if style == "plus91_space":
        return f"+91 {d[:5]} {d[5:]}"
    if style == "plus91_dash":
        return f"+91-{d}"
    if style == "zero":
        return f"0{d[:5]} {d[5:]}"
    if style == "plus91_334":
        return f"+91 {d[:3]} {d[3:6]} {d[6:]}"
    if style == "dash_91":
        return f"91-{d[:5]}-{d[5:]}"
    if style == "spaced_433":
        return f"{d[:4]} {d[4:7]} {d[7:]}"
    if style == "uk":
        return f"+44 7{rand_digits(r, 3)} {rand_digits(r, 6)}"

    # us number
    area = rand_digits(r, 3, r.randint(2, 9))
    exch = rand_digits(r, 3, r.randint(2, 9))
    line = rand_digits(r, 4)
    if style == "us_paren":
        return f"({area}) {exch}-{line}"
    if style == "us_dash":
        return f"{area}-{exch}-{line}"
    if style == "us_plus1":
        return f"+1 {area} {exch} {line}"
    if style == "us_dots":
        return f"{area}.{exch}.{line}"
    raise ValueError(style)


def make_account(r, split):
    d = rand_digits(r, r.randint(11, 16), first=r.randint(1, 9))
    style = r.choice(ACCOUNT_STYLES[split])
    if style == "plain":
        return d
    sep = " " if style == "grouped4" else "-"
    return sep.join(d[i:i + 4] for i in range(0, len(d), 4))


def make_decoy(r, key):
    # these look like phones/accounts but aren't
    if key == "ts":
        return str(r.randint(1_600_000_000, 1_800_000_000))
    if key in ("epoch_ms", "seq"):
        return str(r.randint(1_600_000_000_000, 1_800_000_000_000))
    if key in ("order_id", "booking_ref"):
        return rand_digits(r, 10, first=r.randint(6, 9))  # same shape as a phone, on purpose
    if key in ("txn_ref", "ack_no", "invoice_no"):
        return rand_digits(r, r.randint(12, 15), first=r.randint(1, 9))  # same shape as an account
    if key in ("tracking_no", "awb"):
        return rand_digits(r, 11, first=r.randint(1, 9))
    if key == "build":
        return f"{r.randint(2024, 2026)}.{r.randint(1, 12):02d}.{r.randint(1, 28):02d}.{r.randint(1000, 9999)}"
    if key == "req_id":
        return rand_digits(r, r.randint(8, 12), first=r.randint(1, 9))
    if key == "otp":
        return rand_digits(r, 6, first=r.randint(1, 9))
    raise ValueError(key)


def pick_key(r, kind, split):
    # shifted pool still reuses old keys 40% of the time
    if split == "test" and r.random() < 0.6:
        return r.choice(KEYS["test"][kind])
    return r.choice(KEYS["train"][kind])


def pick_phrase(r, kind, split):
    if split == "test" and r.random() < 0.6:
        return r.choice(PHRASES["test"][kind])
    return r.choice(PHRASES["train"][kind])


def make_fields(r, split):
    fields = []
    for _ in range(r.choice([0, 1, 1, 2])):
        kind = r.choice(["PHONE", "ACCOUNT"])
        value = make_phone(r, split) if kind == "PHONE" else make_account(r, split)
        fields.append((pick_key(r, kind, split), value, kind))
    for _ in range(r.randint(1, 3)):
        key = pick_key(r, "DECOY", split)
        fields.append((key, make_decoy(r, key), None))
    r.shuffle(fields)
    return fields


class Line:
    # builds the text and remembers where the real values are
    def __init__(self):
        self.parts = []
        self.spans = []
        self.pos = 0

    def add(self, text, label=None):
        if label:
            self.spans.append([self.pos, self.pos + len(text), label])
        self.parts.append(text)
        self.pos += len(text)

    def text(self):
        return "".join(self.parts)


def header(r, split):
    level = r.choice(["INFO", "WARN", "DEBUG", "ERROR"])
    svc = r.choice(["payments", "kyc", "support", "orders", "notify"])
    if split == "test" and r.random() < 0.4:
        # syslog style, never seen in train
        return f"<{r.randint(100, 191)}>Oct {r.randint(1, 28)} 13:{r.randint(10, 59)}:{r.randint(10, 59)} host{r.randint(1, 9)} {svc}[{r.randint(100, 9999)}]: "
    return f"2026-10-{r.randint(1, 28):02d}T13:{r.randint(10, 59)}:{r.randint(10, 59)}Z {level} svc={svc} "


def build_kv(r, split):
    line = Line()
    line.add(header(r, split))
    for i, (key, value, label) in enumerate(make_fields(r, split)):
        line.add((", " if i else "") + f"{key}=")
        line.add(value, label)
    return line


def build_json(r, split):
    line = Line()
    head = header(r, split)
    event = r.choice(["update", "signup", "payout", "callback"])
    line.add(head + '{"event":"' + event + '"')
    for key, value, label in make_fields(r, split):
        line.add(f', "{key}": "')
        line.add(value, label)
        line.add('"')
    line.add("}")
    return line


def build_text(r, split):
    # free text, no keys - only the words around the number help
    line = Line()
    line.add(header(r, split) + "note: ")
    pieces = []
    for _ in range(r.choice([0, 1, 1])):
        kind = r.choice(["PHONE", "ACCOUNT"])
        value = make_phone(r, split) if kind == "PHONE" else make_account(r, split)
        pieces.append((pick_phrase(r, kind, split), value, kind))
    for _ in range(r.randint(1, 2)):
        key = pick_key(r, "DECOY", split)
        value = make_decoy(r, key)
        pieces.append((pick_phrase(r, "DECOY", split), value, None))
    r.shuffle(pieces)
    for i, (phrase, value, label) in enumerate(pieces):
        before, after = phrase.split("{v}")
        line.add(("; " if i else "") + before)
        line.add(value, label)
        line.add(after)
    return line


def make_pool(n, split, seed):
    r = random.Random(seed)
    builders = [build_kv, build_json, build_text]
    rows = []
    for _ in range(n):
        line = r.choice(builders)(r, split)
        rows.append({"text": line.text(), "spans": line.spans, "split": split})
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-train", type=int, default=7000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--out", default="raw")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pools = {
        "train_pool.jsonl": make_pool(args.n_train, "train", args.seed),
        "test_pool.jsonl": make_pool(args.n_test, "test", args.seed + 1),
    }
    for name, rows in pools.items():
        with open(out / name, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"{name}: {len(rows)} lines")
