# Verification

- Six unittest integration tests passed: scenarios and SQLite reopening, custom analysis and filtering, malformed/out-of-range inputs, cross-origin POST rejection, failed CSV preserving the active model, and persisted trained-model reload.
- JavaScript syntax check passed (`Get-Content frontend/app.js | node --check`).
- Strict premium UI static audit passed with zero findings (premium-audit.json).
- Browser verified: empty overview, backend connectivity, normal and attack scenario creation, expected edge/fog decisions, saved event history, event modal and Escape focus restoration, keyboard-operated risk filter, no-results state, mobile model registry at 390px, and missing-file validation with focus on the file field.
- Desktop traffic lab and mobile registry visually inspected. Browser activity generated two local demonstration events; the downloadable source archive excludes this database and generated models.
- Real network traffic capture, firewall enforcement, federated clients and research-grade temporal forecasting are outside this build. Detailed limits are in README.md.
- Tested using the Python environment supplied with the original prototype. It emits a SciPy optimization-option warning, but model fitting and tests complete. Fresh dependency installation must be performed when extracting on another computer.

- Official DESIGN.md lint completed with zero errors; six advisory orphan-token warnings reflect CSS-owned values documented in prose.

## Next-move forecasting update

All 15 backend tests pass (`.venv/Scripts/python.exe -m unittest discover -s tests -v`). New checks cover episode/time separation, future-feature exclusion, no cross-episode windows, long-gap exclusion, range/history abstention, failed training preservation, model reload, separate investigations/input sources, saved forecast snapshots, sequence CSV upload, binary-CSV rejection, and the ten-observation demo boundary. SQLite connections now close deterministically; Windows test cleanup verifies files are released.

Both frontend JavaScript files pass Node syntax checks. Strict premium UI audit has zero findings. No design-token changes were made.

Browser checks: first/second observations show remaining history; third shows ranked but inconclusive results; fourth forecast favors credential access; fifth favors lateral movement. These are synthetic demo results, not real-intrusion validation. Also checked disabled controls during requests, preserved investigation after reload, model metrics display, missing sequence-file validation, the corrected Attack scenario button, and a readable forecast panel at 390px width. Existing binary model version 2 remained active throughout verification. Browser verification added six clearly labeled demo/scenario events to the existing local history; earlier events were preserved.

Initial synthetic holdout: 630 training windows from 90 episodes; 210 holdout windows from 30 later episodes; 87.1% accuracy, 72.7% macro F1, 45.2% training-majority baseline. Per-class results are exposed in the UI; recognition of reconnaissance progression is substantially weaker than the aggregate accuracy suggests. No real intrusion sequences were available for evaluation.
