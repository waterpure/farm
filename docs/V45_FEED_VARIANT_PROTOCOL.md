# V45 feed-multiplier v1 protocol

## Question

Does changing only V45 R85's feed-value confidence multiplier improve terminal
cash against the frozen V45 base without making the policy worse against basic
production opponents?

## Candidates

- parent: frozen `v45_base`, multiplier `1.25`;
- challenger A: `1.15`;
- challenger B: `1.35`.

The parent is not modified. Every challenger loads a separate V45 module and
replaces only `_r85_feed`'s multiplier.

## Evaluation

- opponents: `v45_base` (primary), `starter`, `melon`;
- development seeds: `0, 1, 2`;
- holdout seeds: `1000, 1001, 1002`;
- full 720-turn episodes only.

Rank challengers on development mean margin against `v45_base`. A challenger is
ineligible if it has a negative development mean margin against either `starter`
or `melon`; the top eligible challenger alone proceeds to holdout. Report all
three opponent margins, action status, elapsed time and parent telemetry.

This is a local relative comparison, not a leaderboard-score estimate. No
holdout result may be used to choose a new multiplier.

## Result (2026-09-17)

Development selected `1.15` over `1.35` by its less-negative margin against
`v45_base` (`-190.3` versus `-238.7`). The fixed `1.15` holdout margin against
V45 was `+28.0` across three seeds. This does not pass the promotion gate: the
sign is inconsistent with development and the magnitude is negligible relative
to terminal cash. The frozen parent remains the baseline.
