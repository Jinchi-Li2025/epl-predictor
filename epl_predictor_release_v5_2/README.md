# EPL Predictor

**EPL Predictor** is a reliability-first Premier League match forecasting and
match-intelligence platform. It combines statistical models, squad and context
data, guarded LLM summaries, and an interactive Streamlit dashboard.

> Research and educational software only. It is not betting or financial advice.

## Highlights

- Three-way probabilities for home win, draw, and away win
- Expected goals and most likely scorelines
- XGBoost forecasting with a Dixon-Coles score model
- Strict chronological holdout and current-season external validation
- Calibration, Brier Score, Log Loss, RPS, and operational monitoring
- Tactical-board lineup visualisation with predicted and confirmed squads
- Premier League player registry with registration, availability, and season data
- Weather, referee, injuries, team news, and match-importance context
- DeepSeek-generated explanations constrained to structured evidence
- Shadow challengers that cannot replace the production model without validation
- Automated prediction settlement, audit reports, and health checks

## Reliability design

The LLM layer is explanatory only. It does not directly alter production match
probabilities. Candidate model, calibration, and squad features remain in shadow
mode until chronological tests demonstrate a reliable improvement.

Current evaluation outputs are generated locally under `reports/`. They are not
hard-coded into this README because the values change as matches are settled.

## Project structure

```text
app.py                 Streamlit application entry point
pages/                 Model centre, review, player database, and operations UI
src/data/              Match and fixture ingestion
src/features/          Historical, upcoming, and squad feature builders
src/models/            XGBoost, Dixon-Coles, Elo, Poisson, and ensemble logic
src/prediction/        Upcoming match prediction pipeline
src/context/           Weather, availability, lineups, news, and evidence checks
src/evaluation/        Holdout, settlement, monitoring, and challenger gates
src/pipeline/          Daily, matchday, health, and readiness orchestration
src/assets/            Team, stadium, kit, and player-registry builders
configs/               Frozen production configuration and manifests
scripts/               Windows automation and repository checks
```

Generated datasets, reports, model artifacts, logs, backups, downloaded media,
and API keys are intentionally excluded from version control.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -m streamlit run app.py
```

Add only the API keys required by the pipelines you intend to run. Never commit
the real `.env` file.

## Typical update workflow

```powershell
python src\pipeline\run_daily_update.py
python src\pipeline\run_operational_checks.py
python -m streamlit run app.py
```

Some provider plans do not expose future-season lineups, contracts, or market
values. Missing fields remain explicitly unknown rather than being fabricated.

## Repository safety check

Before each public push:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\github_preflight.ps1
```

## Data and attribution

The project can integrate official Premier League squad pages, Fantasy Premier
League data, Football-Data.org, API-Football, public weather services, and other
configured sources. Data availability and reuse conditions remain subject to
each provider's terms. Downloaded crests, kits, stadium photographs, and raw
provider responses are not included in the public source repository by default.

## Model documentation

See [MODEL_CARD.md](MODEL_CARD.md) for model boundaries, validation policy, and
known limitations.

## Status

First public-preview release. The application is operational with non-blocking
warnings while current-season settlement history and squad snapshots accumulate.
