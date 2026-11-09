EPL Predictor v5.1 - Player Profile Integrity Upgrade

Main fixes
1. Canonicalises football-data team names "Tottenham" and "Spurs" to
   "Tottenham Hotspur", fixing the whole-team nationality merge failure.
2. Retrieves and caches detailed football-data team profiles when an API key
   is available. The cache is reused for seven days.
3. Preserves missing public market values as unknown instead of displaying EUR 0.
4. Shows source links and retrieval time for profile, contract and valuation data.
5. Adds optional, source-backed manual overrides without overwriting that file
   during later upgrades.
6. Expands the registry quality audit to detect whole-team nationality gaps and
   report nationality, birth date, contract and market-value coverage.
7. Replaces deprecated Streamlit use_container_width arguments.

Install from the EPL Predictor project root

powershell -ExecutionPolicy Bypass -File ".\epl_predictor_player_profiles_v5_1\install_and_rebuild.ps1"

The rebuild can take a few minutes when detailed team profiles need to be
downloaded. Free football-data accounts are rate limited, so the script waits
between team requests instead of failing.

After installation

python -m streamlit run app.py

Optional verified completion file

data\manual\player_profile_overrides.csv

Only enter values that have a source URL. Separate multiple citizenships with
the | character. Public market values are estimates, not official transfer fees.
