from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import duckdb
import nbformat
from nbclient import NotebookClient
import threading

app = FastAPI()

# Allow CORS for local frontend testing
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

# --- Model List ---
MODELS = [
	{"id": "regression", "name": "Regression (HGB)"},
	{"id": "logistic", "name": "Logistic Regression"},
	{"id": "kmeans", "name": "KMeans Clustering"},
	{"id": "forecast", "name": "Calendar Forecast"},
	{"id": "nlp", "name": "NLP Sentiment Analysis"},
]

@app.get("/api/models")
def get_models():
	return MODELS

# --- Health Check helpers and runner ---
NOTEBOOK_PATH = "WSB-DSS.ipynb"
HEALTH_LOCK = threading.Lock()

def run_health_audit_notebook():
	"""
	Execute notebook cells serially up to the point where data is loaded and
	the health audit can be executed. Strategy:
	  - Run cells from the top, skipping pip installs.
	  - Stop after the last cell that either loads the CSVs or defines run_health_audit/persist_health_audit.
	  - If the notebook doesn't call persist_health_audit itself, inject and run a small snippet that calls
		run_health_audit(listings, calendar, reviews) and persist_health_audit(...).

	Returns True on success.
	"""
	try:
		with HEALTH_LOCK:
			nb = nbformat.read(NOTEBOOK_PATH, as_version=4)

			last_needed_idx = -1

			# 1) Prefer explicit cell metadata tags for deterministic control.
			for i, cell in enumerate(nb.cells):
				tags = cell.get('metadata', {}).get('tags', [])
				if tags:
					for t in tags:
						if isinstance(t, str) and t.lower() == 'ui:health':
							last_needed_idx = max(last_needed_idx, i)
							break

			# 2) If no tag found, use heuristics: search for load/health markers.
			if last_needed_idx < 0:
				load_markers = [
					"listings = pd.read_csv",
					"calendar = pd.read_csv",
					"reviews = pd.read_csv",
					"def run_health_audit",
					"def persist_health_audit",
					"persist_health_audit(",
					"run_health_audit(",
				]
				for i, cell in enumerate(nb.cells):
					src = ''.join(cell.source)
					if src.lstrip().startswith('%pip install'):
						continue
					for marker in load_markers:
						if marker in src:
							last_needed_idx = max(last_needed_idx, i)
							break

			# 3) If we still didn't find markers, fall back to running a safe prefix
			if last_needed_idx < 0:
				last_needed_idx = min(len(nb.cells)-1, 19)

			# Build subset notebook: prefix cells + injected snippet to persist health
			selected_cells = []
			for idx in range(0, last_needed_idx+1):
				cell = nb.cells[idx]
				src = ''.join(cell.source).strip()
				if src.startswith('%pip install'):
					continue
				selected_cells.append(cell)

			post_snippet = """
try:
	audit = run_health_audit(listings, calendar, reviews, verbose=False)
	persist_health_audit(audit, overwrite=True, verbose=False)
	print('WSB_DSS_HEALTH_PERSISTED')
except Exception as e:
	print('WSB_DSS_HEALTH_ERROR', e)
"""

			selected_cells.append(nbformat.v4.new_code_cell(post_snippet))

			subset_nb = nbformat.v4.new_notebook(cells=selected_cells)
			client = NotebookClient(subset_nb, timeout=900, allow_errors=True)
			client.execute()

			# Inspect outputs of the injected snippet (last cell) to detect success or errors
			last_outputs = subset_nb.cells[-1].get('outputs', [])
			stdout_text = ''
			for out in last_outputs:
				# stream outputs (stdout)
				if out.get('output_type') == 'stream':
					stdout_text += out.get('text', '')
				# text/plain or execute_result
				elif out.get('output_type') in ('execute_result', 'display_data'):
					data = out.get('data', {})
					if 'text/plain' in data:
						stdout_text += data['text/plain']
				# error
				elif out.get('output_type') == 'error':
					stdout_text += '\n'.join(out.get('traceback', []))

			# Record whether persisted marker was seen
			persisted = 'WSB_DSS_HEALTH_PERSISTED' in stdout_text

		return True
	except Exception as e:
		print(f"Notebook execution error: {e}")
		return False


@app.post("/api/health/run", response_class=JSONResponse)
def run_and_get_health():
	"""Run notebook cells required for health audit, then return persisted health row."""
	ok = run_health_audit_notebook()
	if not ok:
		return {"status": "error", "error": "Failed to execute health audit cells."}
	try:
		con = duckdb.connect("wsb_dss.duckdb")
		row = con.execute("SELECT * FROM health_checks ORDER BY computed_at DESC LIMIT 1").fetchone()
		if row:
			return {"status": "ok", "data": dict(zip(["dataset_id","computed_at","metrics"], row))}
		else:
			return {"status": "empty", "data": None}
	except Exception as e:
		return {"status": "error", "error": str(e)}

# --- Train Model (stub) ---
@app.post("/api/train/{model_id}")
def train_model(model_id: str):
	return {"status": "started", "model_id": model_id}

# --- Show Artifacts (stub) ---
@app.get("/api/artifacts/{model_id}")
def get_artifacts(model_id: str):
	return {"status": "ok", "model_id": model_id, "artifacts": []}

# --- Serve UI ---
@app.get("/", response_class=HTMLResponse)
def serve_ui(request: Request):
	return HTMLResponse(
		"""
		<!DOCTYPE html>
		<html lang=\"en\">
		<head>
			<meta charset=\"UTF-8\">
			<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
			<title>WSB DSS Dashboard</title>
			<link href=\"https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css\" rel=\"stylesheet\">
			<style>
				body { background: var(--bs-body-bg); color: var(--bs-body-color); }
				.section { margin-bottom: 2rem; }
				.dark-mode { background: #181a1b !important; color: #e0e0e0 !important; }
				.metrics-table td, .metrics-table th { padding: 0.4rem 0.7rem; }
			</style>
		</head>
		<body class=\"bg-light\">
		<div class=\"container py-4\">
			<div class=\"d-flex justify-content-between align-items-center mb-4\">
				<h2>WSB Decision Support System</h2>
				<button id=\"themeToggle\" class=\"btn btn-outline-secondary\">Toggle Theme</button>
			</div>
			<div class=\"row section\">
				<div class=\"col-md-6\">
					<label for=\"modelSelect\" class=\"form-label\">Select Model</label>
					<select id=\"modelSelect\" class=\"form-select\"></select>
				</div>
				<div class=\"col-md-6 d-flex align-items-end\">
					<button id=\"trainBtn\" class=\"btn btn-primary me-2\">Train</button>
					<button id=\"showDataBtn\" class=\"btn btn-outline-info\">Show Data</button>
				</div>
			</div>
			<div class=\"row section\">
				<div class=\"col-12\">
					<h5>Data Health</h5>
					<button id=\"healthBtn\" class=\"btn btn-success mb-2\">Check Health</button>
					<div id=\"healthResult\" class=\"border rounded p-3 bg-white\">No data yet.</div>
				</div>
			</div>
			<div class=\"row section\">
				<div class=\"col-12\">
					<h5>Artifacts & Results</h5>
					<div id=\"artifacts\">No artifacts yet.</div>
				</div>
			</div>
		</div>
		<script src=\"https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js\"></script>
		<script>
		// Theme toggle
		document.getElementById('themeToggle').onclick = function() {
			document.body.classList.toggle('dark-mode');
		};
		// Populate model dropdown
		fetch('/api/models').then(r=>r.json()).then(models => {
			let sel = document.getElementById('modelSelect');
			models.forEach(m => {
				let opt = document.createElement('option');
				opt.value = m.id; opt.textContent = m.name;
				sel.appendChild(opt);
			});
		});
		// Health check (runs notebook cells then fetches persisted health)
		document.getElementById('healthBtn').onclick = function() {
			let res = document.getElementById('healthResult');
			res.textContent = 'Checking...';
			fetch('/api/health/run', {method:'POST'}).then(r=>r.json()).then(d => {
				if(d.status==='ok' && d.data && d.data.metrics) {
					let metrics = {};
					try { metrics = JSON.parse(d.data.metrics); } catch(e) { metrics = d.data.metrics; }
					let html = '<table class="table table-bordered metrics-table">';
					html += '<thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>';
					for (const [k, v] of Object.entries(metrics.rows_cols || {})) {
						html += `<tr><td>${k.replace(/_/g,' ')}</td><td>${v}</td></tr>`;
					}
					if(metrics.duplicates) {
						html += '<tr><th colspan="2">Duplicates</th></tr>';
						for (const [k, v] of Object.entries(metrics.duplicates)) {
							html += `<tr><td>${k.replace(/_/g,' ')}</td><td>${v}</td></tr>`;
						}
					}
					if(metrics.referential) {
						html += '<tr><th colspan="2">Referential Integrity</th></tr>';
						for (const [k, v] of Object.entries(metrics.referential)) {
							html += `<tr><td>${k.replace(/_/g,' ')}</td><td>${v}</td></tr>`;
						}
					}
					if(metrics.date_ranges) {
						html += '<tr><th colspan="2">Date Ranges</th></tr>';
						for (const [k, v] of Object.entries(metrics.date_ranges)) {
							html += `<tr><td>${k.replace(/_/g,' ')}</td><td>${Array.isArray(v) ? v.join(' → ') : v}</td></tr>`;
						}
					}
					if(metrics.review_mismatch_counts) {
						html += '<tr><th colspan="2">Review Mismatch Counts</th></tr>';
						for (const [k, v] of Object.entries(metrics.review_mismatch_counts)) {
							html += `<tr><td>${k.replace(/_/g,' ')}</td><td>${v}</td></tr>`;
						}
					}
					if(metrics.availability_counts) {
						html += '<tr><th colspan="2">Availability Counts</th></tr>';
						for (const [k, v] of Object.entries(metrics.availability_counts)) {
							html += `<tr><td>${k.replace(/_/g,' ')}</td><td>${v}</td></tr>`;
						}
					}
					html += '</tbody></table>';
					res.innerHTML = html;
				} else if(d.status==='ok') {
					res.textContent = 'No health metrics found.';
				} else {
					res.textContent = d.status + (d.error ? ': ' + d.error : '');
				}
			});
		};
		// Train model
		document.getElementById('trainBtn').onclick = function() {
			let model = document.getElementById('modelSelect').value;
			let btn = this;
			btn.disabled = true; btn.textContent = 'Training...';
			fetch('/api/train/' + model, {method:'POST'}).then(r=>r.json()).then(d => {
				btn.disabled = false; btn.textContent = 'Train';
				alert('Training started for ' + model + '. (Stub)');
			});
		};
		// Show artifacts
		document.getElementById('showDataBtn').onclick = function() {
			let model = document.getElementById('modelSelect').value;
			let div = document.getElementById('artifacts');
			div.textContent = 'Loading...';
			fetch('/api/artifacts/' + model).then(r=>r.json()).then(d => {
				if(d.artifacts && d.artifacts.length) {
					div.innerHTML = '<ul>' + d.artifacts.map(a => `<li>${a}</li>`).join('') + '</ul>';
				} else {
					div.textContent = 'No artifacts yet.';
				}
			});
		};
		</script>
		</body>
		</html>
		"""
	)
