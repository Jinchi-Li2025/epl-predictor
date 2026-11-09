# GitHub First-Release Guide

## Recommended repository settings

- Repository name: `epl-predictor`
- Initial visibility: **Private**
- Do not initialise the remote repository with a README, `.gitignore`, or licence
- Enable secret scanning and push protection when available

## GitHub About section

Description:

> A reliability-first Premier League forecasting platform combining XGBoost,
> Dixon-Coles, squad intelligence, contextual data, and guarded LLM explanations
> in an interactive Streamlit dashboard.

Suggested topics:

`premier-league`, `football-analytics`, `machine-learning`, `xgboost`,
`dixon-coles`, `streamlit`, `sports-analytics`, `python`, `llm`, `forecasting`

## Before the first push

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\github_preflight.ps1
git status --short
git ls-files .env
git log --all --oneline -- .env
```

The final two commands must print nothing. If `git ls-files .env` prints `.env`,
stop and run:

```powershell
git rm --cached .env
git commit -m "Remove local environment file from version control"
```

This preserves the local file but removes it from future commits. If a real key
was committed earlier, revoke/rotate the key before publishing; merely deleting
the file from the newest commit does not remove it from history.

For a public repository, downloaded media already tracked by Git can be removed
from version control without deleting the local files:

```powershell
git rm -r --cached --ignore-unmatch assets\stadiums assets\teams\crests assets\teams\kits
```

## Create and push the empty GitHub repository

Create an empty repository on GitHub, then run:

```powershell
git add .
git commit -m "Prepare EPL Predictor v1.0.0 public preview"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/epl-predictor.git
git push -u origin main
```

If `origin` already exists, inspect it first:

```powershell
git remote -v
```

Only if it is the wrong repository, replace it with:

```powershell
git remote set-url origin https://github.com/YOUR-USERNAME/epl-predictor.git
git push -u origin main
```

## First release

- Tag: `v1.0.0`
- Title: `EPL Predictor v1.0.0 - First Public Preview`
- Mark as a pre-release while current-season and squad settlement history grows

Suggested release summary:

> First public preview of EPL Predictor, featuring three-way Premier League
> probability forecasts, expected goals and scorelines, tactical lineup boards,
> a player intelligence database, chronological validation, guarded DeepSeek
> explanations, and operational reliability checks.
