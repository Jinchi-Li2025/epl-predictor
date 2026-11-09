# EPL Predictor Model Card

## Intended use

This project estimates Premier League home/draw/away probabilities, expected
goals, and likely scorelines for research, education, and football analytics.

## Production components

- XGBoost classifier for three-way outcome probabilities
- Dixon-Coles model for expected goals and score distributions
- Frozen feature and production manifests
- Context and squad data displayed as evidence and shadow candidate features

## Validation policy

- Training and evaluation follow chronological boundaries
- Final holdout results remain separate from current-season external validation
- Log Loss, Brier Score, Ranked Probability Score, accuracy, and calibration are
  monitored together
- A challenger is not promoted when its bootstrap confidence interval includes
  no improvement
- LLM output cannot directly change production probabilities or model files

## Current release state

- Operational readiness: usable with non-blocking warnings
- Production calibration challenger: retained in shadow mode
- Current-season external sample: still accumulating
- Squad-feature challenger: waiting for settled pre-match snapshots

Exact current metrics are generated in the local `reports/` directory.

## Important limitations

- Football is intrinsically noisy and low-scoring
- Injuries, late lineup changes, tactical choices, and data revisions can change
  the information available after a prediction is frozen
- Accuracy alone is not sufficient to judge probability quality
- Public player contract and valuation coverage is incomplete and provider-specific
- Market value is an estimate, not an official transfer price

## Prohibited interpretation

The probabilities are not guarantees and should not be treated as betting or
financial advice.
