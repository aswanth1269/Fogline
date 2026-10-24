# UX contract
Visual ownership: DESIGN.md and frontend/style.css.
English interface; dates use the browser locale/timezone with full UTC timestamps in stored events.

## Canonical UI Map
| Capability | Canonical owner | Source of truth | Allowed variants | Verification |
|---|---|---|---|---|
| Select/Listbox | Native select | UX-CONTRACT.md | OS-owned popup | Browser keyboard |
| Form | busy and shared fields in app.js | API validation | Custom input / CSV | API tests and browser |
| Scrollbar | Global style.css | DESIGN.md | Horizontal tables | Narrow browser |
| Toast | toast in app.js | UX-CONTRACT.md | Save / training | Browser status |
| CRUD | API and EventStore | backend/app.py | Create/read only | tests/test_api.py |

## Behavior
Navigation uses hash routes; event filter and page are URL state. Forms remain mounted across route changes so drafts survive navigation. Event data is server paginated, ten rows per page, newest first; filtering resets page. No deletion capability. Event details use native modal dialog with Escape, focus trapping and restoration. API errors remain inline. Duplicate submissions are disabled while pending. Errors do not replace the previous working model. All input labels are associated, invalid fields focus on submit, all status uses a shared live region. Requests are not automatically retried because analysis creates events. Event filter responses use a sequence guard to ignore stale requests. Overview telemetry refreshes every 15 seconds when visible; no traffic is generated automatically. The application is local-only, without login or network enforcement.


## Sequence forecasting
Forecast presentation is owned by frontend/prediction.js and uses existing meter, api, busy, escaping, toast and dialog primitives. The current investigation ID persists locally when browser storage is available. Related manual and demonstration observations are separated server-side; random scenario events never seed forecasts. New investigation changes the ID without deleting data. Forms stay mounted through navigation. Controls that mutate an investigation are disabled during analysis to prevent overlapping submissions. Forecaster training uses the same upload/error/focus conventions as binary training, but calls a separate endpoint. Historical forecast snapshots remain unchanged after retraining. Model details do not rerender while the same model remains active, preserving expanded results across telemetry refreshes.

Forecast states are insufficient history, unrelated samples, out of training range, inconclusive, and ranked forecast. Every state has plain-text explanation. Model scores are explicitly uncalibrated; synthetic provenance remains visible next to the result. Forecasting means next observed event, allowing continuation, not time-to-attack or certainty about intentions. Forecast panels reuse the existing light theme and semantic tokens without a new visual identity.
