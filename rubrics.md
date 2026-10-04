# rubrics

The score doesn't tell the whole story, so I also check these. Each one is pass/fail.

1. **Handles lookalike decoys** - shows it can tell phone-shaped order/booking ids apart from real phones (an example or an error count is enough)
2. **Uses context, not just shape** - predictions depend on the text around the number, not only digit count or layout
3. **Works on shifted logs** - reports normal vs shifted score separately, and the gap is under 0.10
4. **Exact spans** - `+91 98765 43210` comes back as one span, no half numbers or two numbers merged
5. **No leakage** - trains only on `public/`, no test-specific rules, no key lists copied from the test, no outside data
6. **Honest error analysis** - lists the main mistakes left, with counts, including at least one decoy it still gets wrong
