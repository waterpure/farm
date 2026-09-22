# V45 feed-multiplier variants

Parent: the frozen `third_party/v45/main.py` with its retained attribution and
Apache-2.0 notices. This directory describes a local derivative experiment; it
does not grant permission to submit a package or remove source obligations.

## Single change

V45 R85 decides whether to keep feeding an animal by comparing expected product
value with wheat cost. The parent confidence multiplier is `1.25`. The only
variant parameter is that multiplier:

- `1.15`: lower confidence premium, therefore skips feed more readily;
- `1.35`: higher confidence premium, therefore keeps feed more readily.

All routes, market orders, worker logic, safety layers and the parent R85
telemetry remain unchanged. The local loader installs the override after loading
an isolated parent module; it does not edit the frozen source file.

The protocol and results are recorded in `docs/V45_FEED_VARIANT_PROTOCOL.md`
and `experiments/v45_feed_multiplier_v1.jsonl`.

## Result

This experiment did not promote either challenger. `1.15` was the better
development candidate but trailed frozen V45 by 190.3 mean terminal cash; its
three-seed holdout led by only 28.0. That sign change and tiny effect relative to
episode cash make the result inconclusive, not an improvement.
