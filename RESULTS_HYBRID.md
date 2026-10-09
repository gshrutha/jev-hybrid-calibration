# Reproduction and hybrid calibration of Jev on unseen support tickets

This fork of [scienthoon/jev-ood-calibration](https://github.com/scienthoon/jev-ood-calibration) does two things:

1. Reproduces the original calibration study of TypeSafe's Jev on 900 synthetic support-ticket questions.
2. Tests whether per-question-type calibration and a small hybrid model (Jev's answers combined with structured ticket fields) fix the miscalibration the original study found.

## 1. Reproduction (8 October 2026)

Setup is identical to the original, with one change: `zeroDataRetention` is set to `false` in `scripts/jev_eval.mjs`, because Vercel's Hobby plan does not support Zero Data Retention. The data is synthetic, so this has no privacy implications and should not affect Jev's outputs.

Model `typesafe-ai/jev` via Vercel AI Gateway, 900 items, 0 failed calls, about $0.015.

| | Original (19 Sep 2026) | Reproduction (8 Oct 2026) |
|---|---|---|
| Overall accuracy | 75.1% | 74.7% |
| Overall ECE | 0.107 (4.4× noise floor) | 0.114 (4.6× noise floor) |
| Refit temperature (global) | 2.74 | 2.74 |
| Queue (choice): accuracy / ECE | 89.0% / 0.082 | 89.0% / 0.083 |
| Angry (noul): accuracy / ECE | 91.7% / 0.079 | 92.0% / 0.076 |
| Priority (score): accuracy / ECE | 44.7% / 0.325 | 43.0% / 0.338 |
| `confidence` field ECE | 0.18 | 0.186 |

The main findings reproduce. Priority accuracy moved by about 5 tickets between runs. Jev is described as deterministic, so either the model was updated between the two dates or its outputs are not fully deterministic; this is untested.

Full output: `results/my_run.jsonl` (raw responses) and `results/my_run_eval.txt` (metrics).

## 2. Per-type calibration and a hybrid model

`scripts/hybrid_calib.py` compares five approaches using 5-fold cross-validation over the 300 tickets, so every result is measured on tickets the method was not fitted on:

- **raw**: Jev's probabilities as returned (exact zeros clipped to 1e-4 for all methods, so NLL values differ slightly from `eval_dump`).
- **global T**: one temperature for all question types.
- **per-type T**: one temperature per question type.
- **hybrid (Jev)**: L2-regularised logistic regression predicting priority from Jev's three answers for the same ticket (queue, angry, priority). It uses Jev's predictions only, never the true queue or angry labels.
- **hybrid (+meta)**: the same, plus `customer_tier` and `channel`.

| Method | Type | Accuracy | NLL | ECE |
|---|---|---|---|---|
| raw | all | 0.747 | 0.672 | 0.114 |
| global T | all | 0.747 | 0.574 | 0.098 |
| per-type T | all | 0.747 | 0.530 | 0.072 |
| raw | priority | 0.430 | 1.246 | 0.338 |
| per-type T | priority | 0.430 | 1.002 | 0.147 |
| hybrid (Jev) | priority | 0.660 | 0.749 | 0.045 |
| hybrid (+meta) | priority | 0.743 | 0.644 | 0.066 |
| **per-type T + hybrid (+meta)** | **all** | **0.851** | **0.410** | **0.032** |

Mean per-type temperatures across folds: choice 2.19, noul 0.66, score 2.57.

Full output: `results/hybrid_results.txt`.

## Findings

1. **A single global temperature hurts some question types.** The yes/no questions need sharpening (T ≈ 0.66) while choice and score need softening (T > 2). A global temperature worsens yes/no ECE from 0.076 to 0.195. Per-type temperatures cut overall ECE from 0.114 to 0.072.
2. **Calibration cannot fix accuracy.** With its own temperature, priority remains at 43% accuracy.
3. **Jev knows the parts but does not combine them.** Learning a combination of Jev's own three answers raises priority accuracy from 43% to 66%, with ECE 0.045. Jev judges anger correctly 92% of the time, and anger is part of the priority rule, yet its own priority answer does not use this.
4. **Structured fields add further signal.** Adding customer tier raises priority accuracy to 74% with the best NLL. The ECE difference between the two hybrids is within noise at n = 300.

## Limitations

- Synthetic data with a known rule; the tier effect is built in by design.
- The hybrid needs labelled examples (240 tickets per fold); zero-shot Jev does not.
- One cross-validation split, n = 300 per question type, no confidence intervals yet.
- Priority stays well below the ~95% ceiling (5% label noise), because each template's base urgency is not fully captured by Jev's answers or tier.
- The logistic regression is a MAP estimate; it does not represent uncertainty about its own parameters.

## Next steps

- Repeat across cross-validation seeds and freshly generated ticket sets, with bootstrap confidence intervals.
- Learning curve: hybrid performance with 20, 50 and 100 labelled tickets.
- Fully Bayesian version (Laplace approximation or MCMC).
- Richer text features from an LLM to capture template-level urgency.

## Reproduce

```
node scripts/jev_eval.mjs --in data/val.jsonl --out results/my_run.jsonl --model typesafe-ai/jev --concurrency 8
python -m scripts.eval_dump --dump results/my_run.jsonl --fit-temperature --confidence-reliability
python scripts/hybrid_calib.py --data data/val.jsonl --results results/my_run.jsonl
```

## Credit

Original study, code and data by [scienthoon](https://github.com/scienthoon/jev-ood-calibration) (code MIT, data CC0). Reproduction, `scripts/hybrid_calib.py` and this write-up by G Shrutha Keerthi Reddy, under the same MIT licence.
