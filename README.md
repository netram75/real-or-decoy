# real-or-decoy

Find the real phone and bank account numbers in messy server logs, and ignore the numbers that only look like them.

I built this as a small ML challenge: synthetic data generator, public/private split, grader, a shortcut baseline, a reference solution and rubrics.

## The task

Each row is one log line. Some lines have real identifiers:

- `PHONE` - Indian, US and UK numbers in lots of layouts
- `ACCOUNT` - bank account numbers, 11 to 16 digits, plain or grouped

Every line also has decoys: timestamps, order ids, tracking numbers, reference numbers, build numbers, OTPs. Many of them are the exact same shape as a phone or account number.

You need to return the exact span and label of every real one.

```
2026-10-23T13:34:52Z ERROR svc=support note: order 36846563212233 shipped; reach me on 082917 03423; ticket 1643295767015 closed
```

Only `082917 03423` is real here (PHONE). The other two are decoys.

## Files

- `generate_data.py` - makes the logs (random + seeded, no real data)
- `prepare.py` - splits into `public/` and `private/`
- `grade.py` - scores a submission
- `baseline.py` - labels numbers by shape only
- `solution.py` - reference model, uses the text around each number
- `rubrics.md` - how to judge a solution beyond the score

Public files:
- `train.csv` - `id, text, spans` (spans = JSON list of `[start, end, label]`)
- `test.csv` - `id, text`
- `sample_submission.csv` - `id, start, end, label`

## Scoring

One row per predicted number: `id, start, end, label` (end is exclusive, same as python slicing).

A prediction counts only if span and label match exactly. Score = macro F1 over PHONE and ACCOUNT. The grader rejects broken files (unknown ids, bad labels, spans outside the text).

## Run it

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python generate_data.py --seed 7
.venv/bin/python prepare.py --seed 7
.venv/bin/python solution.py public submission.csv
.venv/bin/python grade.py --submission submission.csv --by-split
```

## Results

| | score | normal lines | shifted lines |
|---|---|---|---|
| baseline (shape only) | 0.507 | 0.521 | 0.499 |
| solution.py | **0.887** | 0.991 | 0.833 |

The test set has 1000 lines that look like the training data and 2000 "shifted" lines: new key names (`msisdn`, `payee_acc`, `ack_no`...), new layouts (`+91 987 654 3210`, `415.555.0132`, `+44 7xxx xxxxxx`, dashed account numbers), new decoys (`awb`, `booking_ref`, `seq`, `otp`) and a syslog style header that never shows up in train.

## Why it's hard

- Decoys have the same shape on purpose. Order and booking ids are 10 digits starting with 6-9, exactly like an Indian mobile. Reference numbers are 12-15 digits, like account numbers.
- So the label comes from context, and the context changes in the hidden test.
- Spans have to be exact - `+91 98765 43210` is one span including the prefix and spaces.
- Free text lines have no keys at all: "ring 98765 43210 after 6" vs "booking 9876543210 confirmed".

## Shortcuts that don't work

- "10 digits = phone" also catches phone-shaped order ids. Baseline phone precision is 0.43.
- "11-16 digits = account" catches every reference number. Account precision is 0.27.
- Memorising key names gets 0.99 on normal lines but drops to 0.83 on shifted ones.

## Where the reference model still fails

From the errors on shifted lines:

- `ack_no` is a decoy but gets called an account 109 times - it looks like `acc`.
- phone-shaped `booking_ref` and `awb` values get called phones.
- it misses new account keys (`payee_acc`, `savings_acc`, `ac_no`) and phrases like "deposit into ... (savings)".

So there's still room above 0.833 on the shifted part.

## Rules if you try it

- train from scratch on every run, only on `public/`
- no hardcoded test answers or rules written for the test
- no external data, standard python ML libs only
- fixed seed, results should reproduce

All data here is randomly generated. No real people or accounts.
