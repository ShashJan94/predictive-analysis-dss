## Notebook cell map (WSB-DSS.ipynb)

Below is a cell-by-cell map of `WSB-DSS.ipynb`. Each item lists the cell index (1-based), a short summary of what that cell contains, and the primary functions or side-effects defined there. This makes it easy to find the health-audit and model functions used by the API.

1. Cell 1 — %pip install duckdb
	- Installs `duckdb` (quiet pip install). The FastAPI runner intentionally skips `%pip install` lines when executing the notebook programmatically.

2. Cell 2 — Imports & setup
	- Standard imports (pandas, numpy, sklearn helpers, matplotlib, warnings, display options). Prepares environment variables and pandas display settings.

3. Cell 3 — %pip install kaggle
	- Installs `kaggle` (quiet pip install). Also skipped when the API executes the notebook subset.

4. Cell 4 — Kaggle dataset download helper
	- Uses `KaggleApi` and a local `kaggle.json` (under `kaggle_key/`) to download `listings.csv`, `calendar.csv`, and `reviews.csv` into the `data/airbnb_seattle` folder if they are missing.

5. Cell 5 — CSV checks and data load
	- Implements `check_csv()` to sanity-check required columns. Loads `listings`, `calendar`, and `reviews` into DataFrame variables and parses date columns. Prints dataset shapes.

6. Cell 6 — Run tracking & artifact DB helpers
	- Defines DuckDB path, `artifacts` root and DB helpers: `_con()`, `start_run()`, `end_run()`, `log_artifact()`, `latest_run()`, `runs_history()`.

7. Cell 7 — Audit tables & persist helpers
	- Creates singleton tables for `health_checks` and `deep_dive_checks`. Defines `persist_health_audit()` and `persist_deep_dive_audit()` that write audit JSON and detail tables into DuckDB.

8. Cell 8 — Health audit function
	- `run_health_audit(listings, calendar, reviews, verbose=True)` — core health check implementation producing a `{'metrics':..., 'tables':...}` dict. Computes shapes, missingness, duplicates, referential integrity, date ranges, price summaries and returns dataframes used by persistors.

9. Cell 9 — Deep-dive audit function
	- `run_deep_dive_audit(...)` — deeper analysis (occupancy, gaps, review stats, neighborhood summaries) that returns structured `metrics` and `tables` for deep inspection.

10. Cell 10 — Run & persist audits
	 - Example calls that run `run_health_audit(...)` and `persist_health_audit(...)`, then the deep-dive equivalent, and prints what was stored. Useful as a one-click run when working interactively.

11. Cell 11 — Regression training (HGB)
	 - `train_regression_hgb(listings_df, reviews_df, ...)` — trains a HistGradientBoostingRegressor on log(price) and returns model, metrics, feature importance and predictions. Contains feature engineering and plotting options.

12. Cell 12 — Logistic and KMeans utilities
	 - `run_logistic_price_bucket(...)` — logistic pipeline for price-bucket classification and metrics/plots.
	 - `run_kmeans_clusters(...)` — KMeans clustering helper and summaries.

13. Cell 13 — Calendar forecaster
	 - `run_calendar_forecast(calendar_df, ...)` — builds time-series features from calendar, trains an HGB forecaster, evaluates vs seasonal naive, and produces future forecasts + plots and holdout tables.

14. Cell 14 — NLP sentiment scoring
	 - `run_review_sentiment_nlp(reviews_df, ...)` — multilingual sentiment scoring (transformer fallback to VADER), returns scored reviews and rollups (per-listing summaries, monthly trend, overall%); also saves a small `overall_pct.json` artifact when persisted.

15. Cell 15 — Filesystem helpers + persistors for models + examples
	 - Filesystem helpers (save figures, outdir creation) and `persist_*_outputs` helpers for `regression`, `logistic`, `kmeans`, `forecast`, and `nlp` that save model artifacts, metrics and tables under `artifacts/<model_type>/<run_id>/` and log them in DuckDB. Contains minimal example end-to-end runs (calls training functions and persists outputs). Running this cell will create artifacts and DB rows.

16. Cell 16 — Empty / placeholder
	 - Blank cell at the end of the notebook (safe to ignore).

Notes about the notebook and the API
- The FastAPI `app.py` uses heuristics and tags to decide which cells to execute when `/api/health/run` is called:
  - It looks for cell metadata tags `ui:health` (case-insensitive) and will include tagged cells deterministically.
  - If no tags are present, it searches for load markers like `listings = pd.read_csv`, `def run_health_audit`, `persist_health_audit(` and runs up to the latest cell containing those markers.
  - It automatically skips `%pip install` cells.
  - The API injects a small snippet (calls `run_health_audit(...)` + `persist_health_audit(...)`) and runs the subset with `nbclient` so the web UI can call the endpoint without re-running the entire notebook.

## How to use and modify the notebook & repo

Read this section before editing the notebook or the API behavior.

  1. Open `WSB-DSS.ipynb` in Jupyter or VS Code Notebook editor.
  2. Run cells in order (or run the cells you modify). The example calls in Cell 15 will execute training and persist artifacts — comment them out if you do not want to run full experiments.

  - Start the API: `uvicorn app:app --reload --host 127.0.0.1 --port 8000`.
  - POST to `/api/health/run` (from the UI press "Check Health" or run `curl -X POST http://127.0.0.1:8000/api/health/run`). The app will run a conservative subset of the notebook and read the latest row from `wsb_dss.duckdb`.


## Suggestions for `app.py` and calling specific notebook cells from the UI

Below are safe, practical suggestions for improving `app.py` so the UI (or callers) can request that the server execute a specific set of notebook cells (by index), or a named tag. I include a small recommended server-side contract and example front-end JS and PowerShell requests you can use right away.

Why this is useful
- Running only a small set of notebook cells reduces runtime and risk when the notebook contains heavy training cells.
- Allowing the UI to request specific cell indexes (or tags) makes the system predictable for reproducible health checks and developer testing.

Design / contract suggestions (server-side)
- Endpoint: Keep `POST /api/health/run` but accept an optional JSON body with either:
	- `cell_indexes`: list of 1-based integer indexes (e.g., [2,5,7]) — OR —
	- `tags`: list of metadata tag strings to include (e.g., ["ui:health"]).
- Behavior:
	- If `cell_indexes` provided: validate indexes (must be ints between 1 and number of cells). Convert to 0-based when selecting cells.
	- Else if `tags` provided: include any cells whose metadata tags match (case-insensitive).
	- Else: fall back to the current heuristic behavior (existing logic in `run_health_audit_notebook`).
	- Always skip `%pip install` lines and never run cells that look like external installs.
	- Optionally: enforce a whitelist of allowed indexes/tags (recommended in production) to prevent abuse.

Security & safety notes
- Do not allow arbitrary cells that perform heavy computations or external network calls unless you trust the caller.
- Consider adding simple authentication for the endpoint (e.g., environment-token checked in a header) before allowing `cell_indexes` execution.
- Log the selected indexes/tags and the user (if authenticated) for auditability.

Suggested API request examples

- Request using JSON body (POST) — call from PowerShell:

```powershell
$body = @{ cell_indexes = @(2,5,7) } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/health/run -Body $body -ContentType 'application/json'
```

- Request using JSON body (fetch from UI JS):

```javascript
fetch('/api/health/run', {
	method: 'POST',
	headers: { 'Content-Type': 'application/json' },
	body: JSON.stringify({ cell_indexes: [2,5,7] })
}).then(r => r.json()).then(console.log).catch(console.error)
```

Example: simple UI change
- Add an input on the served HTML to accept a comma-separated list of cell indexes and pass it as JSON to `/api/health/run`. Example minimal change to the HTML's health button handler:

```javascript
// read indexes from an input with id 'cellIndexesInput' (e.g., "2,5,7")
const raw = document.getElementById('cellIndexesInput')?.value || '';
const idxs = raw.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
fetch('/api/health/run', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({cell_indexes: idxs}) })
	.then(r => r.json()).then(d => { console.log(d); /* update UI */ });
```

Suggested server-side implementation sketch (safe, minimal change)
- Modify `run_health_audit_notebook` to accept optional parameters `cell_indexes=None, tags=None` and make the selection procedure explicit:
	1. Read notebook with `nbformat.read()`.
	2. If `cell_indexes` is provided: convert to 0-based list and select only those cells (preserve order).
	3. Else if `tags` provided: include all cells whose metadata tags contain any of the requested tags.
	4. Else: use existing heuristic to pick the prefix.
	5. As a final step append the injected snippet that calls `run_health_audit(...)` + `persist_health_audit(...)` if not already persisted by the notebook.

Small pseudo-code (Python) to guide the change:

```python
def run_health_audit_notebook(cell_indexes: list[int]|None=None, tags: list[str]|None=None):
		nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
		if cell_indexes:
				# convert 1-based (external) to 0-based
				sel = [nb.cells[i-1] for i in cell_indexes if 1 <= i <= len(nb.cells)]
		elif tags:
				sel = [c for c in nb.cells if any(t.lower() in (x.lower() for x in c.get('metadata', {}).get('tags', [])) for t in tags)]
		else:
				# existing heuristic code that finds last_needed_idx
				sel = nb.cells[:last_needed_idx+1]
		# skip %pip install cells and append injected snippet
		filtered = [c for c in sel if not ''.join(c.get('source','')).lstrip().startswith('%pip install')]
		filtered.append(nbformat.v4.new_code_cell(post_snippet))
		subset_nb = nbformat.v4.new_notebook(cells=filtered)
		client = NotebookClient(subset_nb, timeout=900, allow_errors=True)
		client.execute()
		# inspect outputs and return
```

Notes about indexes vs tags
- Indexes are deterministic but fragile: inserting or removing notebook cells changes indexes and UI callers must be updated.
- Tags are stable and recommended for long-term use; for example add `ui:health` to cells required for the health run and then the UI can request the `tags=['ui:health']` option.

Automated change offer
- I can implement the server-side change now (add optional `cell_indexes`/`tags` parsing to `POST /api/health/run` and wire it to `run_health_audit_notebook`). It's a small, localized change and I will include:
	- Input validation (bounds checking and size limit, e.g., max 20 indexes),
	- Logging the requested indexes/tags,
	- A short note in the UI that shows which indexes were used.


You can also tag the jupyter notebook cells and use the metadata to hook with the API and send commands from the UI. 


