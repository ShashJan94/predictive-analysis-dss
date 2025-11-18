from fastapi import FastAPI, Request, HTTPException, Body
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import duckdb
import nbformat
from nbclient import NotebookClient
import threading
import os
import json
import copy
from pathlib import Path
from typing import List, Dict, Any
from urllib.parse import quote

import pandas as pd

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
NOTEBOOK_FILE = BASE_DIR / "WSB-DSS.ipynb"
DATA_DIR = BASE_DIR / "data" / "airbnb_seattle"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Ensure artifact directory exists so downstream listing calls do not fail
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# Simple "run a single cell" debug endpoint (kept, but UI will no longer use it)
# -------------------------------------------------------------------------
@app.post("/api/run_cell/{cell_index}")
def run_notebook_cell(cell_index: int):
    """Run a single code cell by index in the notebook and return its output."""
    try:
        with NOTEBOOK_FILE.open('r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)

        if cell_index < 0 or cell_index >= len(nb.cells):
            return {"status": "error", "error": "Invalid cell index"}

        cell = nb.cells[cell_index]
        if cell['cell_type'] != 'code':
            return {"status": "skipped", "message": "Cell is not a code cell."}

        temp_nb = nbformat.v4.new_notebook()
        temp_nb.cells = [cell]

        client = NotebookClient(temp_nb, timeout=60, kernel_name='python3', allow_errors=True)
        client.execute()

        outputs = temp_nb.cells[0].get('outputs', [])
        result = []
        for output in outputs:
            if 'text' in output:
                result.append(output['text'])
            elif 'data' in output and 'text/plain' in output['data']:
                result.append(output['data']['text/plain'])
            elif output.get('output_type') == 'error':
                result.append('\n'.join(output.get('traceback', [])))

        return {"status": "ok", "outputs": result}
    except Exception as e:
        return {"status": "error", "error": str(e)}


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


# IMPORTANT: we **do not** include cell 3 (Kaggle) here.
MODEL_TRAIN_CELLS: Dict[str, List[int]] = {
    "regression": [1, 4, 5, 6, 11, 15, 16],
    "logistic":   [1, 4, 5, 6, 11, 12, 15, 17],
    "kmeans":     [1, 4, 5, 6, 11, 12, 15, 18],
    "forecast":   [1, 4, 5, 6, 13, 15, 19],
    "nlp":        [1, 4, 5, 6, 14, 15, 20],
}

MODEL_NOTEBOOK_LOCK = threading.Lock()

DATA_FILES = {
    "listings": DATA_DIR / "listings.csv",
    "calendar": DATA_DIR / "calendar.csv",
    "reviews": DATA_DIR / "reviews.csv",
}


def execute_notebook_cells(cell_indexes: List[int], *, timeout: int = 900) -> Dict[str, Any]:
    """Execute selected notebook code cells sequentially and capture outputs in ONE kernel."""
    if not cell_indexes:
        return {"status": "ok", "results": [], "errors": []}

    with NOTEBOOK_FILE.open('r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    selected_cells = []
    executed_indices: List[int] = []
    for idx in cell_indexes:
        if idx < 0 or idx >= len(nb.cells):
            raise ValueError(f"Invalid cell index: {idx}")
        cell = nb.cells[idx]
        if cell.get('cell_type') != 'code':
            continue
        source = ''.join(cell.get('source', '')).strip()
        if source.startswith('%pip '):
            continue
        selected_cells.append(copy.deepcopy(cell))
        executed_indices.append(idx)

    if not selected_cells:
        return {"status": "ok", "results": [], "errors": []}

    temp_nb = nbformat.v4.new_notebook(cells=selected_cells)
    client = NotebookClient(temp_nb, timeout=timeout, allow_errors=True)
    client.execute()

    results = []
    errors = []
    for original_idx, cell in zip(executed_indices, temp_nb.cells):
        cell_outputs = []
        for out in cell.get('outputs', []):
            otype = out.get('output_type')
            if otype == 'error':
                errors.append({
                    "cell_index": original_idx,
                    "ename": out.get('ename'),
                    "evalue": out.get('evalue'),
                    "traceback": out.get('traceback', []),
                })
            elif otype == 'stream':
                cell_outputs.append(out.get('text', ''))
            elif otype in {'execute_result', 'display_data'}:
                data = out.get('data', {})
                text_value = data.get('text/plain')
                if text_value:
                    cell_outputs.append(text_value)
        results.append({"cell_index": original_idx, "outputs": cell_outputs})

    status = "ok" if not errors else "error"
    return {"status": status, "results": results, "errors": errors}


# NEW: run a list of cells in one kernel (this is what the UI uses now)
@app.post("/api/run_cells", response_class=JSONResponse)
def run_cells(payload: Dict[str, Any] = Body(...)):
    """
    Execute a list of notebook cells in a single kernel and return outputs.
    Payload: { "cells": [1,4,5,...] }
    """
    cells = payload.get("cells") or []
    if not isinstance(cells, list):
        raise HTTPException(status_code=400, detail="Body must contain a 'cells' list.")
    try:
        # Optional lock so we don't overlap training & health runs
        with MODEL_NOTEBOOK_LOCK:
            result = execute_notebook_cells([int(c) for c in cells])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        return {"status": "error", "error": str(exc)}
    return result


def build_artifact_payload(model_id: str) -> Dict[str, Any]:
    model_dir = ARTIFACTS_DIR / model_id
    if not model_dir.exists() or not model_dir.is_dir():
        return {}

    candidate_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
    if not candidate_dirs:
        return {}
    latest_dir = max(candidate_dirs, key=lambda p: p.stat().st_mtime)

    metrics_path = latest_dir / "metrics.json"
    metrics: Dict[str, Any] = {}
    if metrics_path.exists():
        with metrics_path.open('r', encoding='utf-8') as f:
            metrics = json.load(f)

    plot_files = sorted(
        [f for f in latest_dir.iterdir() if f.is_file() and f.suffix.lower() in {'.png', '.jpg', '.jpeg', '.svg'}]
    )
    other_files = sorted([f for f in latest_dir.iterdir() if f.is_file() and f.name not in {p.name for p in plot_files}])

    def _make_url(path: Path) -> str:
        return f"/artifacts/file/{quote(model_id)}/{quote(latest_dir.name)}/{quote(path.name)}"

    plots = [{"name": p.name, "url": _make_url(p)} for p in plot_files]
    files = [{"name": f.name, "url": _make_url(f)} for f in other_files if f.name != 'metrics.json']

    return {
        "model_id": model_id,
        "run_id": latest_dir.name,
        "metrics": metrics,
        "plots": plots,
        "files": files,
    }


def load_csv_preview(limit: int = 5) -> Dict[str, Any]:
    preview: Dict[str, Any] = {}
    for label, path in DATA_FILES.items():
        if not path.exists():
            preview[label] = {"columns": [], "rows": []}
            continue
        df = pd.read_csv(path, nrows=limit)
        preview[label] = {
            "columns": list(df.columns),
            "rows": df.fillna('').to_dict(orient='records'),
        }
    return preview


# --- Health Check helpers and runner ---
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
            nb = nbformat.read(str(NOTEBOOK_FILE), as_version=4)

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
                last_needed_idx = min(len(nb.cells) - 1, 19)

            # Build subset notebook: prefix cells + injected snippet to persist health
            selected_cells = []
            for idx in range(0, last_needed_idx + 1):
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
            return {"status": "ok", "data": dict(zip(["dataset_id", "computed_at", "metrics"], row))}
        else:
            return {"status": "empty", "data": None}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/api/health/latest", response_class=JSONResponse)
def get_latest_health():
    """Return the latest health_checks row without executing the notebook."""
    try:
        con = duckdb.connect("wsb_dss.duckdb")
        row = con.execute("SELECT * FROM health_checks ORDER BY computed_at DESC LIMIT 1").fetchone()
        if row:
            return {"status": "ok", "data": dict(zip(["dataset_id", "computed_at", "metrics"], row))}
        else:
            return {"status": "empty", "data": None}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/api/deepdive/latest", response_class=JSONResponse)
def get_latest_deepdive():
    """Return the latest deep_dive_checks row without executing the notebook."""
    try:
        con = duckdb.connect("wsb_dss.duckdb")
        row = con.execute("SELECT * FROM deep_dive_checks ORDER BY computed_at DESC LIMIT 1").fetchone()
        if row:
            return {"status": "ok", "data": dict(zip(["dataset_id", "computed_at", "metrics"], row))}
        else:
            return {"status": "empty", "data": None}
    except Exception as e:
        return {"status": "error", "error": str(e)}


DEEPDIVE_DETAIL_TABLES = {
    "miss_by_avail": "detail_deepdive_price_missing_by_avail",
    "occ_by_listing": "detail_deepdive_occupancy_by_listing",
    "gap_summary": "detail_deepdive_gap_summary",
    "notable_gaps": "detail_deepdive_notable_gaps",
    "rev_stats": "detail_deepdive_review_stats",
    "top_neighborhoods": "detail_deepdive_top_neighborhoods",
}


@app.get("/api/deepdive/tables", response_class=JSONResponse)
def get_deepdive_tables(dataset_id: str = "airbnb_seattle", limit: int = 10):
    """Return preview rows for each deep dive detail table."""
    try:
        con = duckdb.connect("wsb_dss.duckdb")
        tables_out = {}
        for key, table_name in DEEPDIVE_DETAIL_TABLES.items():
            # ensure table exists before querying
            exists_row = con.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
                [table_name]
            ).fetchone()
            if not exists_row or exists_row[0] == 0:
                continue
            query = f"SELECT * FROM {table_name} WHERE dataset_id = ? LIMIT ?"
            cursor = con.execute(query, [dataset_id, limit])
            description = cursor.description or []
            columns = [col[0] for col in description]
            rows = cursor.fetchall()
            tables_out[key] = {
                "columns": columns,
                "rows": [dict(zip(columns, row)) for row in rows],
            }
        return {"status": "ok", "tables": tables_out}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/api/data/sample", response_class=JSONResponse)
def show_data_sample(limit: int = 5):
    """Return a small preview for each source CSV."""
    try:
        preview = load_csv_preview(limit=limit)
        return {"status": "ok", "data": preview}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


@app.post("/api/train/{model_id}", response_class=JSONResponse)
def train_model(model_id: str):
    """Execute the notebook cells necessary to train the selected model."""
    cells = MODEL_TRAIN_CELLS.get(model_id)
    if not cells:
        raise HTTPException(status_code=404, detail=f"Unknown model id '{model_id}'")

    try:
        with MODEL_NOTEBOOK_LOCK:
            result = execute_notebook_cells(cells)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        return {"status": "error", "error": str(exc)}

    if result.get("status") != "ok":
        return {"status": "error", "error": "Notebook execution failed", "details": result.get("errors", [])}

    payload = build_artifact_payload(model_id)
    response_body = {
        "status": "ok",
        "model_id": model_id,
        "logs": result.get("results", []),
    }
    if payload:
        response_body["run"] = payload
    return response_body


@app.get("/api/artifacts/{model_id}", response_class=JSONResponse)
def get_artifacts(model_id: str):
    """Return latest metrics and artifact metadata for the selected model."""
    if model_id not in MODEL_TRAIN_CELLS:
        raise HTTPException(status_code=404, detail=f"Unknown model id '{model_id}'")
    payload = build_artifact_payload(model_id)
    if not payload:
        return {"status": "empty", "model_id": model_id}
    return {"status": "ok", "model_id": model_id, "run": payload}


@app.get("/artifacts/file/{model_id}/{run_id}/{filename}")
def serve_artifact_file(model_id: str, run_id: str, filename: str):
    """Serve artifact files (plots, metrics, tables) for display or download."""
    base_dir = (ARTIFACTS_DIR / model_id / run_id).resolve()
    file_path = (base_dir / filename).resolve()
    if not str(file_path).startswith(str(base_dir)) or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found.")
    return FileResponse(file_path)


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
        <div class="container py-4">
            <div class="d-flex justify-content-between align-items-center mb-4">
                <h2>WSB Decision Support System</h2>
                <button id="themeToggle" class="btn btn-outline-secondary">Toggle Theme</button>
            </div>
            <div class="row section">
                <div class="col-md-6">
                    <label for="modelSelect" class="form-label">Select Model</label>
                    <select id="modelSelect" class="form-select"></select>
                </div>
                <div class="col-md-6 d-flex align-items-end justify-content-end flex-wrap gap-2">
                    <button id="trainBtn" class="btn btn-primary">Train</button>
                </div>
            </div>
            <div class="row section">
                <div class="col-12">
                    <h5>Dataset Preview</h5>
                    <div class="d-flex flex-wrap gap-2 mb-3">
                        <button id="showDataBtn" class="btn btn-outline-info">Show Data</button>
                    </div>
                    <div id="dataPreview" class="border rounded p-3 bg-white d-none">No preview loaded yet.</div>
                </div>
            </div>
            <div class="row section">
                <div class="col-12">
                    <h5>Data Health</h5>
                    <div class="d-flex flex-wrap gap-2 mb-3">
                        <button id="healthBtn" class="btn btn-success">Check Health</button>
                        <button id="deepDiveBtn" class="btn btn-warning">Deep Dive</button>
                        <button id="healthToggle" class="btn btn-outline-secondary">Hide Health</button>
                        <button id="deepDiveToggle" class="btn btn-outline-secondary">Hide Deep Dive</button>
                        <button id="artifactsToggle" class="btn btn-outline-secondary">Hide Results</button>
                    </div>
                    <div id="healthResult" class="border rounded p-3 bg-white">No data yet.</div>
                    <div id="deepDiveResult" class="border rounded p-3 bg-white mt-3">No data yet.</div>
                    <h5 class="mt-4" id="artifactsLabel">Model Outputs</h5>
                    <div id="artifacts" class="border rounded p-3 bg-white">Select a model and run training to see results.</div>
                    <div id="trainLogs" class="border rounded p-3 bg-white mt-3">No logs yet.</div>
                </div>
            </div>
        </div>
        <div id="bootstrapStatus" class="visually-hidden"></div>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
        <script>
        document.addEventListener('DOMContentLoaded', () => {
            const themeToggle = document.getElementById('themeToggle');
            const modelSelect = document.getElementById('modelSelect');
            const trainBtn = document.getElementById('trainBtn');
            const showDataBtn = document.getElementById('showDataBtn');
            const dataPreviewDiv = document.getElementById('dataPreview');
            const healthResultDiv = document.getElementById('healthResult');
            const deepDiveResultDiv = document.getElementById('deepDiveResult');
            const artifactsDiv = document.getElementById('artifacts');
            const artifactsLabel = document.getElementById('artifactsLabel');
            const trainLogsDiv = document.getElementById('trainLogs');
            const bootstrapStatus = document.getElementById('bootstrapStatus');

            let currentModel = null;

            // IMPORTANT: these cell indexes are run together in ONE kernel via /api/run_cells
            const DEFAULT_BOOTSTRAP_CELLS = [1,4,5,6];          // imports + data load + helpers
            const HEALTH_CELLS           = [1,4,5,6,7,9];       // health audit path (no Kaggle)
            const DEEP_DIVE_CELLS       = [1,4,5,6,8,10];      // deep dive path (no Kaggle)

            const DEEP_DIVE_TABLE_LABELS = {
                miss_by_avail: 'Price missingness by availability',
                occ_by_listing: 'Occupancy by listing',
                gap_summary: 'Gap summary',
                notable_gaps: 'Notable gaps',
                rev_stats: 'Review stats',
                top_neighborhoods: 'Top neighborhoods',
            };

            const ESCAPE_MAP = {
                '&': '&amp;',
                '<': '&lt;',
                '>': '&gt;',
                '"': '&quot;',
                "'": '&#39;',
            };

            function escapeHtml(value) {
                if (value === null || value === undefined) {
                    return '';
                }
                return String(value).replace(/[&<>'"]/g, (char) => ESCAPE_MAP[char] || char);
            }

            function toggleSection(btnId, sectionId, hideText, showText) {
                const btn = document.getElementById(btnId);
                const section = document.getElementById(sectionId);
                if (!btn || !section) {
                    return;
                }
                btn.addEventListener('click', () => {
                    const currentlyHidden = section.classList.toggle('d-none');
                    btn.textContent = currentlyHidden ? showText : hideText;
                });
            }

            // NEW: uses /api/run_cells so all requested cells share one kernel
            async function runCellsSequentially(cellIndexes, statusDivId) {
                const statusDiv = document.getElementById(statusDivId);
                if (!statusDiv) {
                    return false;
                }
                if (!cellIndexes || !cellIndexes.length) {
                    statusDiv.textContent = 'No cells to run.';
                    return true;
                }
                statusDiv.textContent = 'Running cells...';
                try {
                    const resp = await fetch('/api/run_cells', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ cells: cellIndexes }),
                    });
                    const data = await resp.json();
                    if (data.status !== 'ok') {
                        statusDiv.textContent = data.error || 'Error while running notebook cells.';
                        return false;
                    }
                    statusDiv.textContent = 'All cells have run successfully!';
                    return true;
                } catch (err) {
                    statusDiv.textContent = 'Unexpected error while running notebook cells.';
                    return false;
                }
            }

            function flattenMetrics(prefix, value, rows) {
                if (value && typeof value === 'object' && !Array.isArray(value)) {
                    for (const [key, val] of Object.entries(value)) {
                        const nextPrefix = prefix ? `${prefix} → ${key}` : key;
                        flattenMetrics(nextPrefix, val, rows);
                    }
                    return;
                }
                rows.push({ key: prefix, value });
            }

            function formatMetricValue(value) {
                if (value === null || value === undefined) {
                    return '';
                }
                if (typeof value === 'number') {
                    return Number.isFinite(value) ? value.toLocaleString(undefined, { maximumFractionDigits: 4 }) : String(value);
                }
                if (Array.isArray(value)) {
                    return value.map(formatMetricValue).join(', ');
                }
                return String(value);
            }

            function renderMetrics(metrics) {
                if (!metrics || Object.keys(metrics).length === 0) {
                    return '<p class="text-muted mb-3">No metrics available.</p>';
                }
                const rows = [];
                for (const [key, value] of Object.entries(metrics)) {
                    flattenMetrics(key, value, rows);
                }
                let html = '<div class="mb-3"><h6>Metrics</h6><div class="table-responsive"><table class="table table-sm table-bordered"><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>';
                rows.forEach(row => {
                    html += `<tr><td>${escapeHtml(row.key.replace(/_/g, ' '))}</td><td>${escapeHtml(formatMetricValue(row.value))}</td></tr>`;
                });
                html += '</tbody></table></div></div>';
                return html;
            }

            function renderPlots(plots) {
                if (!plots || plots.length === 0) {
                    return '';
                }
                let html = '<div class="mb-3"><h6>Plots</h6><div class="row g-3">';
                plots.forEach(plot => {
                    html += `<div class="col-sm-6 col-lg-4"><img src="${plot.url}" alt="${escapeHtml(plot.name)}" class="img-fluid border rounded"></div>`;
                });
                html += '</div></div>';
                return html;
            }

            function renderFiles(files) {
                if (!files || files.length === 0) {
                    return '';
                }
                let html = '<div class="mb-3"><h6>Artifacts</h6><ul class="list-unstyled mb-0">';
                files.forEach(file => {
                    html += `<li><a href="${file.url}" target="_blank" rel="noopener">${escapeHtml(file.name)}</a></li>`;
                });
                html += '</ul></div>';
                return html;
            }

            function renderRun(run) {
                if (!run) {
                    artifactsDiv.textContent = 'No artifacts yet.';
                    return;
                }
                let html = `<p><strong>Latest run:</strong> ${escapeHtml(run.run_id || 'unknown')}</p>`;
                html += renderMetrics(run.metrics || {});
                html += renderPlots(run.plots || []);
                html += renderFiles(run.files || []);
                if ((!run.metrics || Object.keys(run.metrics).length === 0) && (!run.plots || run.plots.length === 0) && (!run.files || run.files.length === 0)) {
                    html += '<p class="text-muted mb-0">No saved outputs for this run.</p>';
                }
                artifactsDiv.innerHTML = html;
                artifactsLabel.textContent = 'Model Outputs';
            }

            function renderLogs(logs) {
                if (!logs || logs.length === 0) {
                    trainLogsDiv.textContent = 'No logs yet.';
                    return;
                }
                let html = '<h6>Notebook Outputs</h6>';
                logs.forEach(entry => {
                    const cellLabel = (entry.cell_index !== undefined && entry.cell_index !== null)
                        ? entry.cell_index
                        : '?';
                    const outputHtml = (entry.outputs && entry.outputs.length)
                        ? entry.outputs.map(out => escapeHtml(out).replace(/\\n/g, '<br>')).join('<br>')
                        : '<span class="text-muted">No output.</span>';
                    html += `<details class="mb-2"><summary>Cell ${escapeHtml(cellLabel)}</summary><div class="bg-light border rounded p-2 small">${outputHtml}</div></details>`;
                });
                trainLogsDiv.innerHTML = html;
            }

            function renderErrorDetails(errors) {
                if (!errors || errors.length === 0) {
                    trainLogsDiv.innerHTML = '<p class="text-danger mb-0">No additional error details.</p>';
                    return;
                }
                let html = '<h6>Errors</h6>';
                errors.forEach(err => {
                    const cellLabel = (err.cell_index !== undefined && err.cell_index !== null)
                        ? err.cell_index
                        : '?';
                    const message = err.evalue || err.error || 'Execution error';
                    html += `<div class="mb-2"><strong>Cell ${escapeHtml(cellLabel)}</strong>`;
                    html += `<div class="bg-light border rounded p-2 small text-danger mb-2">${escapeHtml(message)}</div>`;
                    if (err.traceback && err.traceback.length) {
                        html += `<pre class="bg-dark text-white small p-2 rounded">${escapeHtml(err.traceback.join('\\n'))}</pre>`;
                    }
                    html += '</div>';
                });
                trainLogsDiv.innerHTML = html;
            }

            function renderDataPreview(preview) {
                if (!preview || Object.keys(preview).length === 0) {
                    dataPreviewDiv.textContent = 'No preview available.';
                    return;
                }
                let html = '';
                for (const [name, info] of Object.entries(preview)) {
                    const title = name.replace(/_/g, ' ');
                    if (!info.rows || info.rows.length === 0) {
                        html += `<p class="mb-3"><strong>${escapeHtml(title)}:</strong> No rows.</p>`;
                        continue;
                    }
                    html += `<div class="mb-3"><h6 class="text-capitalize">${escapeHtml(title)}</h6>`;
                    html += '<div class="table-responsive"><table class="table table-sm table-bordered"><thead><tr>';
                    info.columns.forEach(col => {
                        html += `<th>${escapeHtml(col)}</th>`;
                    });
                    html += '</tr></thead><tbody>';
                    info.rows.forEach(row => {
                        html += '<tr>';
                        info.columns.forEach(col => {
                            const value = row[col];
                            html += `<td>${escapeHtml(value == null ? '' : value)}</td>`;
                        });
                        html += '</tr>';
                    });
                    html += '</tbody></table></div></div>';
                }
                dataPreviewDiv.innerHTML = html;
            }

            function renderTablePreview(label, table) {
                if (!table || !table.rows || table.rows.length === 0) {
                    return `<p class="mb-3"><strong>${escapeHtml(label)}:</strong> No rows.</p>`;
                }
                const columns = table.columns.filter(col => col !== 'dataset_id');
                let html = `<div class="mb-3"><h6>${escapeHtml(label)}</h6><div class="table-responsive"><table class="table table-sm table-bordered"><thead><tr>`;
                columns.forEach(col => {
                    html += `<th>${escapeHtml(col.replace(/_/g, ' '))}</th>`;
                });
                html += '</tr></thead><tbody>';
                table.rows.forEach(row => {
                    html += '<tr>';
                    columns.forEach(col => {
                        const value = row[col];
                        html += `<td>${escapeHtml(value == null ? '' : value)}</td>`;
                    });
                    html += '</tr>';
                });
                html += '</tbody></table></div></div>';
                return html;
            }

            async function loadArtifacts(modelId) {
                if (!modelId) {
                    return;
                }
                artifactsDiv.textContent = 'Loading latest results...';
                trainLogsDiv.textContent = 'No logs yet.';
                artifactsLabel.textContent = 'Model Outputs';
                try {
                    const response = await fetch(`/api/artifacts/${modelId}`);
                    const data = await response.json();
                    if (data.status === 'ok' && data.run) {
                        renderRun(data.run);
                    } else if (data.status === 'empty') {
                        artifactsDiv.textContent = 'No artifacts yet.';
                    } else {
                        artifactsDiv.innerHTML = `<p class="text-danger mb-0">${escapeHtml(data.error || 'Unable to load artifacts.')}</p>`;
                    }
                } catch (err) {
                    artifactsDiv.innerHTML = '<p class="text-danger mb-0">Failed to load artifacts.</p>';
                }
            }

            async function loadModels() {
                try {
                    const models = await fetch('/api/models').then(r => r.json());
                    modelSelect.innerHTML = '';
                    models.forEach(model => {
                        const opt = document.createElement('option');
                        opt.value = model.id;
                        opt.textContent = model.name;
                        modelSelect.appendChild(opt);
                    });
                    if (models.length) {
                        currentModel = models[0].id;
                        modelSelect.value = currentModel;
                        await loadArtifacts(currentModel);
                    }
                } catch (err) {
                    artifactsDiv.innerHTML = '<p class="text-danger mb-0">Failed to load models.</p>';
                }
            }

            themeToggle.addEventListener('click', () => {
                document.body.classList.toggle('dark-mode');
            });

            toggleSection('healthToggle', 'healthResult', 'Hide Health', 'Show Health');
            toggleSection('deepDiveToggle', 'deepDiveResult', 'Hide Deep Dive', 'Show Deep Dive');
            toggleSection('artifactsToggle', 'artifacts', 'Hide Results', 'Show Results');

            modelSelect.addEventListener('change', async () => {
                currentModel = modelSelect.value;
                await loadArtifacts(currentModel);
            });

            // --- Show Data toggle (one button: Show ↔ Hide) ---
            let dataLoaded = false;
            let dataVisible = false;

            showDataBtn.addEventListener('click', async () => {
                if (!dataVisible) {
                    const previousText = showDataBtn.textContent;
                    showDataBtn.disabled = true;
                    showDataBtn.textContent = 'Loading...';

                    if (!dataLoaded) {
                        dataPreviewDiv.classList.remove('d-none');
                        dataPreviewDiv.textContent = 'Loading...';
                        try {
                            const response = await fetch('/api/data/sample?limit=5');
                            const data = await response.json();
                            if (data.status === 'ok') {
                                renderDataPreview(data.data || {});
                                dataLoaded = true;
                            } else {
                                dataPreviewDiv.innerHTML =
                                    `<p class="text-danger mb-0">${escapeHtml(data.error || 'Failed to load data preview.')}</p>`;
                            }
                        } catch (err) {
                            dataPreviewDiv.innerHTML =
                                '<p class="text-danger mb-0">Unexpected error while loading data.</p>';
                        } finally {
                            showDataBtn.disabled = false;
                            showDataBtn.textContent = 'Hide Data';
                            dataVisible = true;
                        }
                    } else {
                        dataPreviewDiv.classList.remove('d-none');
                        showDataBtn.disabled = false;
                        showDataBtn.textContent = 'Hide Data';
                        dataVisible = true;
                    }
                } else {
                    dataPreviewDiv.classList.add('d-none');
                    showDataBtn.textContent = 'Show Data';
                    dataVisible = false;
                }
            });

            // --- Health & Deep Dive buttons using /api/run_cells ---
            document.getElementById('healthBtn').addEventListener('click', async () => {
                const ok = await runCellsSequentially(HEALTH_CELLS, 'healthResult');
                if (!ok) {
                    return;
                }
                healthResultDiv.textContent = 'Fetching health metrics...';
                try {
                    const data = await fetch('/api/health/latest').then(r => r.json());
                    if (data.status === 'ok' && data.data && data.data.metrics) {
                        let metrics = {};
                        try { metrics = JSON.parse(data.data.metrics); } catch (err) { metrics = data.data.metrics; }
                        let html = '<table class="table table-bordered metrics-table">';
                        html += '<thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>';
                        for (const [k, v] of Object.entries(metrics.rows_cols || {})) {
                            html += `<tr><td>${escapeHtml(k.replace(/_/g, ' '))}</td><td>${escapeHtml(v)}</td></tr>`;
                        }
                        if (metrics.duplicates) {
                            html += '<tr><th colspan="2">Duplicates</th></tr>';
                            for (const [k, v] of Object.entries(metrics.duplicates)) {
                                html += `<tr><td>${escapeHtml(k.replace(/_/g, ' '))}</td><td>${escapeHtml(v)}</td></tr>`;
                            }
                        }
                        if (metrics.referential) {
                            html += '<tr><th colspan="2">Referential Integrity</th></tr>';
                            for (const [k, v] of Object.entries(metrics.referential)) {
                                html += `<tr><td>${escapeHtml(k.replace(/_/g, ' '))}</td><td>${escapeHtml(v)}</td></tr>`;
                            }
                        }
                        if (metrics.date_ranges) {
                            html += '<tr><th colspan="2">Date Ranges</th></tr>';
                            for (const [k, v] of Object.entries(metrics.date_ranges)) {
                                const value = Array.isArray(v) ? v.join(' \u2192 ') : v;
                                html += `<tr><td>${escapeHtml(k.replace(/_/g, ' '))}</td><td>${escapeHtml(value)}</td></tr>`;
                            }
                        }
                        if (metrics.review_mismatch_counts) {
                            html += '<tr><th colspan="2">Review Mismatch Counts</th></tr>';
                            for (const [k, v] of Object.entries(metrics.review_mismatch_counts)) {
                                html += `<tr><td>${escapeHtml(k.replace(/_/g, ' '))}</td><td>${escapeHtml(v)}</td></tr>`;
                            }
                        }
                        if (metrics.availability_counts) {
                            html += '<tr><th colspan="2">Availability Counts</th></tr>';
                            for (const [k, v] of Object.entries(metrics.availability_counts)) {
                                html += `<tr><td>${escapeHtml(k.replace(/_/g, ' '))}</td><td>${escapeHtml(v)}</td></tr>`;
                            }
                        }
                        html += '</tbody></table>';
                        healthResultDiv.innerHTML = html;
                    } else if (data.status === 'ok') {
                        healthResultDiv.textContent = 'No health metrics found.';
                    } else {
                        healthResultDiv.textContent = data.status + (data.error ? ': ' + data.error : '');
                    }
                } catch (err) {
                    healthResultDiv.textContent = 'Failed to load health metrics.';
                }
            });

            document.getElementById('deepDiveBtn').addEventListener('click', async () => {
                const ok = await runCellsSequentially(DEEP_DIVE_CELLS, 'deepDiveResult');
                if (!ok) {
                    return;
                }
                deepDiveResultDiv.textContent = 'Fetching deep dive metrics...';
                try {
                    const data = await fetch('/api/deepdive/latest').then(r => r.json());
                    if (data.status === 'ok' && data.data && data.data.metrics) {
                        let metrics = {};
                        try { metrics = JSON.parse(data.data.metrics); } catch (err) { metrics = data.data.metrics; }
                        let html = '<h6>Summary metrics</h6><table class="table table-bordered metrics-table">';
                        html += '<thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>';
                        if (metrics.occupancy) {
                            for (const [k, v] of Object.entries(metrics.occupancy)) {
                                html += `<tr><td>${escapeHtml('occupancy ' + k)}</td><td>${escapeHtml(formatMetricValue(v))}</td></tr>`;
                            }
                        }
                        if (metrics.gaps_max !== undefined) {
                            html += `<tr><td>gaps max</td><td>${escapeHtml(formatMetricValue(metrics.gaps_max))}</td></tr>`;
                        }
                        if (metrics.share_with_review !== undefined && metrics.share_with_review !== null) {
                            html += `<tr><td>share with \u22651 review</td><td>${escapeHtml(((metrics.share_with_review * 100) || 0).toFixed(2))}%</td></tr>`;
                        }
                        if (metrics.median_prices) {
                            for (const [k, v] of Object.entries(metrics.median_prices)) {
                                html += `<tr><td>${escapeHtml('median price ' + k)}</td><td>${escapeHtml(formatMetricValue(v))}</td></tr>`;
                            }
                        }
                        if (metrics.neigh_col) {
                            html += `<tr><td>neighborhood column</td><td>${escapeHtml(metrics.neigh_col)}</td></tr>`;
                        }
                        html += '</tbody></table>';
                        const tablesResp = await fetch('/api/deepdive/tables?limit=10');
                        const tablesJson = await tablesResp.json();
                        if (tablesJson.status === 'ok' && tablesJson.tables) {
                            html += '<hr><h6>Detail tables (first 10 rows)</h6>';
                            for (const [key, table] of Object.entries(tablesJson.tables)) {
                                const label = DEEP_DIVE_TABLE_LABELS[key] || key;
                                html += renderTablePreview(label, table);
                            }
                        }
                        deepDiveResultDiv.innerHTML = html;
                    } else if (data.status === 'ok') {
                        deepDiveResultDiv.textContent = 'No deep dive metrics found.';
                    } else {
                        deepDiveResultDiv.textContent = data.status + (data.error ? ': ' + data.error : '');
                    }
                } catch (err) {
                    deepDiveResultDiv.textContent = 'Failed to load deep dive metrics.';
                }
            });

            trainBtn.addEventListener('click', async () => {
                if (!modelSelect.value) {
                    return;
                }
                const originalText = trainBtn.textContent;
                trainBtn.disabled = true;
                trainBtn.textContent = 'Training...';
                artifactsDiv.innerHTML = '<p class="text-muted mb-0">Executing notebook cells...</p>';
                trainLogsDiv.textContent = 'Awaiting notebook logs...';
                try {
                    const response = await fetch(`/api/train/${modelSelect.value}`, { method: 'POST' });
                    const data = await response.json();
                    if (data.status === 'ok') {
                        if (data.run) {
                            renderRun(data.run);
                        } else {
                            artifactsDiv.innerHTML = '<p class="text-muted mb-0">Training completed, but no artifacts were produced.</p>';
                        }
                        renderLogs(data.logs || []);
                        if (!data.run) {
                            await loadArtifacts(modelSelect.value);
                        }
                    } else {
                        artifactsDiv.innerHTML = `<p class="text-danger mb-0">${escapeHtml(data.error || 'Training failed.')}</p>`;
                        renderErrorDetails(data.details || []);
                    }
                } catch (err) {
                    artifactsDiv.innerHTML = '<p class="text-danger mb-0">Unexpected error while training.</p>';
                    renderErrorDetails([]);
                } finally {
                    trainBtn.disabled = false;
                    trainBtn.textContent = originalText;
                }
            });

            loadModels();
            runCellsSequentially(DEFAULT_BOOTSTRAP_CELLS, 'bootstrapStatus').catch(() => {});
        });
        </script>
        </body>
        </html>
        """
    )
