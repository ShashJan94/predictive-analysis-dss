document.addEventListener('DOMContentLoaded', () => {
			const themeToggle = document.getElementById('themeToggle');
			const modelSelect = document.getElementById('modelSelect');
			const trainBtn = document.getElementById('trainBtn');
			const showDataBtn = document.getElementById('showDataBtn');
			const healthBtn = document.getElementById('healthBtn');
			const deepDiveBtn = document.getElementById('deepDiveBtn');
			const dataPreviewDiv = document.getElementById('dataPreview');
			const healthResultDiv = document.getElementById('healthResult');
			const deepDiveResultDiv = document.getElementById('deepDiveResult');
			const artifactsDiv = document.getElementById('artifacts');
			const artifactsLabel = document.getElementById('artifactsLabel');
			const trainLogsDiv = document.getElementById('trainLogs');

			const state = { currentModel: null };

			const ESCAPE_MAP = {
				'&': '&amp;',
				'<': '&lt;',
				'>': '&gt;',
				'"': '&quot;',
				"'": '&#39;',
			};

			const DEEP_DIVE_TABLE_LABELS = {
				miss_by_avail: 'Price missingness by availability',
				occ_by_listing: 'Occupancy by listing',
				gap_summary: 'Gap summary',
				notable_gaps: 'Notable gaps',
				rev_stats: 'Review stats',
				top_neighborhoods: 'Top neighborhoods',
			};

			function escapeHtml(value) {
				if (value === null || value === undefined) {
					return '';
				}
				return String(value).replace(/[&<>'"]/g, (char) => ESCAPE_MAP[char] || char);
			}

			function setText(element, text) {
				if (element) {
					element.textContent = text;
				}
			}

			function toggleSection(buttonId, sectionId, hideText, showText) {
				const button = document.getElementById(buttonId);
				const section = document.getElementById(sectionId);
				if (!button || !section) {
					return;
				}
				button.addEventListener('click', () => {
					const hidden = section.classList.toggle('d-none');
					button.textContent = hidden ? showText : hideText;
				});
			}

			async function requestJson(url, options = {}) {
				const response = await fetch(url, options);
				let data = null;
				try {
					data = await response.json();
				} catch (err) {
					return { ok: response.ok, data: null, error: 'Invalid JSON response' };
				}
				if (!response.ok) {
					const message = data && data.error ? data.error : response.statusText || 'Request failed';
					return { ok: false, data, error: message };
				}
				return { ok: true, data };
			}

			function renderArtifacts(run) {
				if (!run) {
					artifactsDiv.textContent = 'No artifacts yet.';
					return;
				}
				let html = `<p><strong>Latest run:</strong> ${escapeHtml(run.run_id || 'unknown')}</p>`;
				if (run.metrics && Object.keys(run.metrics).length) {
					html += '<h6 class="mt-3">Metrics</h6>';
					html += `<pre class="bg-light border rounded p-2 small">${escapeHtml(JSON.stringify(run.metrics, null, 2))}</pre>`;
				}
				if (Array.isArray(run.plots) && run.plots.length) {
					html += '<h6 class="mt-3">Plots</h6><div class="row g-3">';
					run.plots.forEach((plot) => {
					if (!plot || !plot.url) {
						return;
					}
					const url = escapeHtml(plot.url);
					const name = plot.name ? escapeHtml(plot.name) : 'Plot';
					html += `<div class="col-sm-6 col-lg-4"><img src="${url}" alt="${name}" class="img-fluid border rounded"></div>`;
					});
					html += '</div>';
				}
				if (Array.isArray(run.files) && run.files.length) {
					html += '<h6 class="mt-3">Files</h6><ul class="list-unstyled mb-0">';
					run.files.forEach((file) => {
					if (!file || !file.url) {
						return;
					}
					const url = escapeHtml(file.url);
					const name = file.name ? escapeHtml(file.name) : 'Download';
					html += `<li><a href="${url}" target="_blank" rel="noopener">${name}</a></li>`;
					});
					html += '</ul>';
				}
				if ((!run.metrics || !Object.keys(run.metrics).length) && (!run.plots || !run.plots.length) && (!run.files || !run.files.length)) {
					html += '<p class="text-muted mb-0">No saved outputs for this run.</p>';
				}
				artifactsDiv.innerHTML = html;
				artifactsLabel.textContent = 'Model Outputs';
			}

			function renderLogs(logs) {
				if (!Array.isArray(logs) || !logs.length) {
					trainLogsDiv.textContent = 'No logs yet.';
					return;
				}
				let html = '<h6>Notebook Outputs</h6>';
				logs.forEach((entry, index) => {
					const label = entry && entry.cell_index !== undefined ? entry.cell_index : index;
					const outputs = entry && Array.isArray(entry.outputs) ? entry.outputs.join('\n') : '';
					html += `<details class="mb-2"><summary>Cell ${escapeHtml(String(label))}</summary>`;
					html += outputs ? `<pre class="bg-light border rounded p-2 small">${escapeHtml(outputs)}</pre>` : '<p class="text-muted small mb-0">No output.</p>';
					html += '</details>';
				});
				trainLogsDiv.innerHTML = html;
			}

			function renderErrors(errors) {
				if (!Array.isArray(errors) || !errors.length) {
					trainLogsDiv.innerHTML = '<p class="text-danger mb-0">No additional notebook details.</p>';
					return;
				}
				let html = '<h6>Notebook Errors</h6>';
				errors.forEach((err, index) => {
					const label = err && err.cell_index !== undefined ? err.cell_index : index;
					const message = err && (err.evalue || err.error) ? err.evalue || err.error : 'Execution error';
					html += `<div class="mb-2"><strong>Cell ${escapeHtml(String(label))}</strong>`;
					html += `<div class="bg-light border rounded p-2 small text-danger mb-2">${escapeHtml(message)}</div>`;
					if (err && Array.isArray(err.traceback) && err.traceback.length) {
						html += `<pre class="bg-dark text-white small p-2 rounded">${escapeHtml(err.traceback.join('\n'))}</pre>`;
					}
					html += '</div>';
				});
				trainLogsDiv.innerHTML = html;
			}

			function renderDataPreview(preview) {
				const entries = Object.entries(preview || {});
				if (!entries.length) {
					dataPreviewDiv.textContent = 'No preview available.';
					return;
				}
				let html = '';
				entries.forEach(([name, info]) => {
					const title = name.replace(/_/g, ' ');
					const infoObj = info || {};
					const columns = Array.isArray(infoObj.columns) ? infoObj.columns : [];
					const rows = Array.isArray(infoObj.rows) ? infoObj.rows : [];
					html += `<div class="mb-3"><h6 class="text-capitalize">${escapeHtml(title)}</h6>`;
					if (!rows.length || !columns.length) {
						html += '<p class="text-muted mb-0">No rows.</p></div>';
						return;
					}
					html += '<div class="table-responsive"><table class="table table-sm table-bordered"><thead><tr>';
					columns.forEach((col) => {
						html += `<th>${escapeHtml(col)}</th>`;
					});
					html += '</tr></thead><tbody>';
					rows.forEach((row) => {
						html += '<tr>';
						columns.forEach((col) => {
							const value = row && row[col] != null ? row[col] : '';
							html += `<td>${escapeHtml(value)}</td>`;
						});
						html += '</tr>';
					});
					html += '</tbody></table></div></div>';
				});
				dataPreviewDiv.innerHTML = html;
			}

			function renderDetailTable(key, table) {
				const label = DEEP_DIVE_TABLE_LABELS[key] || key.replace(/_/g, ' ');
				const info = table || {};
				const rawColumns = Array.isArray(info.columns) ? info.columns : [];
				const columns = rawColumns.filter((col) => col !== 'dataset_id');
				const rows = Array.isArray(info.rows) ? info.rows : [];
				let html = `<div class="mb-3"><h6>${escapeHtml(label)}</h6>`;
				if (!rows.length || !columns.length) {
					html += '<p class="text-muted mb-0">No rows.</p></div>';
					return html;
				}
				html += '<div class="table-responsive"><table class="table table-sm table-bordered"><thead><tr>';
				columns.forEach((col) => {
					html += `<th>${escapeHtml(col.replace(/_/g, ' '))}</th>`;
				});
				html += '</tr></thead><tbody>';
				rows.forEach((row) => {
					html += '<tr>';
					columns.forEach((col) => {
						const value = row && row[col] != null ? row[col] : '';
						html += `<td>${escapeHtml(value)}</td>`;
					});
					html += '</tr>';
				});
				html += '</tbody></table></div></div>';
				return html;
			}

			function renderDetailTables(tables) {
				const entries = Object.entries(tables || {});
				if (!entries.length) {
					return '<p class="text-muted mb-0">No detail tables available.</p>';
				}
				let html = '';
				entries.forEach(([key, table]) => {
					html += renderDetailTable(key, table);
				});
				return html;
			}

			async function refreshArtifacts() {
				if (!state.currentModel) {
					artifactsDiv.textContent = 'Select a model to see artifacts.';
					return;
				}
				setText(artifactsDiv, 'Loading latest results...');
				const { ok, data, error } = await requestJson(`/api/artifacts/${state.currentModel}`);
				if (!ok) {
					artifactsDiv.innerHTML = `<p class="text-danger mb-0">${escapeHtml(error || 'Unable to load artifacts.')}</p>`;
					return;
				}
				if (data.status === 'empty') {
					artifactsDiv.textContent = 'No artifacts yet.';
					return;
				}
				if (data.status !== 'ok') {
					artifactsDiv.innerHTML = `<p class="text-danger mb-0">${escapeHtml(data.error || 'Unable to load artifacts.')}</p>`;
					return;
				}
				renderArtifacts(data.run);
			}

			async function loadModels() {
				const { ok, data, error } = await requestJson('/api/models');
				if (!ok || !Array.isArray(data)) {
					artifactsDiv.innerHTML = `<p class="text-danger mb-0">${escapeHtml(error || 'Failed to load models.')}</p>`;
					return;
				}
				modelSelect.innerHTML = '';
				data.forEach((model) => {
					if (!model || !model.id) {
						return;
					}
					const option = document.createElement('option');
					option.value = model.id;
					option.textContent = model.name || model.id;
					modelSelect.appendChild(option);
				});
				if (data.length) {
					state.currentModel = data[0].id;
					modelSelect.value = state.currentModel;
					await refreshArtifacts();
				}
			}

			async function runNotebookCells(indexes, statusElement) {
				if (!Array.isArray(indexes) || !indexes.length) {
					return true;
				}
				if (statusElement) {
					statusElement.textContent = 'Running notebook cells...';
				}
				for (let i = 0; i < indexes.length; i += 1) {
					const index = indexes[i];
					if (statusElement) {
						statusElement.textContent = `Running cell ${index} (${i + 1}/${indexes.length})...`;
					}
					const { ok, data, error } = await requestJson(`/api/run_cell/${index}`, { method: 'POST' });
					if (!ok || (data && data.status && data.status !== 'ok' && data.status !== 'skipped')) {
						if (statusElement) {
							const message = error || (data && data.error) || 'Execution error.';
							statusElement.textContent = `Error in cell ${index}: ${message}`;
						}
						return false;
					}
				}
				if (statusElement) {
					statusElement.textContent = 'Notebook cells finished.';
				}
				return true;
			}

			async function handleDataPreview() {
				const originalText = showDataBtn.textContent;
				showDataBtn.disabled = true;
				showDataBtn.textContent = 'Loading...';
				setText(dataPreviewDiv, 'Loading...');
				const { ok, data, error } = await requestJson('/api/data/sample?limit=5');
				if (ok && data && data.status === 'ok') {
					renderDataPreview(data.data || {});
				} else {
					dataPreviewDiv.innerHTML = `<p class="text-danger mb-0">${escapeHtml(error || (data && data.error) || 'Failed to load data preview.')}</p>`;
				}
				showDataBtn.disabled = false;
				showDataBtn.textContent = originalText;
			}

			async function handleHealth() {
				const ran = await runNotebookCells([0, 1, 2, 3, 4, 5, 6, 7, 9], healthResultDiv);
				if (!ran) {
					return;
				}
				setText(healthResultDiv, 'Fetching health metrics...');
				const { ok, data, error } = await requestJson('/api/health/latest');
				if (!ok) {
					healthResultDiv.textContent = `Failed to load health metrics: ${error}`;
					return;
				}
				if (data.status === 'ok' && data.data && data.data.metrics) {
					const metrics = data.data.metrics;
					const content = typeof metrics === 'string' ? metrics : JSON.stringify(metrics, null, 2);
					healthResultDiv.innerHTML = `<pre class="bg-light border rounded p-2 small">${escapeHtml(content)}</pre>`;
				} else if (data.status === 'empty') {
					healthResultDiv.textContent = 'No health metrics found.';
				} else {
					healthResultDiv.textContent = data.error ? `Error: ${data.error}` : 'Health metrics unavailable.';
				}
			}

			async function handleDeepDive() {
				const ran = await runNotebookCells([0, 1, 2, 3, 4, 5, 6, 7, 8, 10], deepDiveResultDiv);
				if (!ran) {
					return;
				}
				setText(deepDiveResultDiv, 'Fetching deep dive metrics...');
				const latest = await requestJson('/api/deepdive/latest');
				let html = '';
				if (!latest.ok) {
					html += `<p class="text-danger mb-3">${escapeHtml(latest.error || 'Unable to load deep dive metrics.')}</p>`;
				} else if (latest.data.status === 'ok' && latest.data.data && latest.data.data.metrics) {
					const metrics = latest.data.data.metrics;
					const metricsText = typeof metrics === 'string' ? metrics : JSON.stringify(metrics, null, 2);
					html += `<pre class="bg-light border rounded p-2 small mb-3">${escapeHtml(metricsText)}</pre>`;
				} else if (latest.data.status === 'empty') {
					html += '<p class="text-muted mb-3">No deep dive metrics found.</p>';
				} else {
					html += `<p class="text-danger mb-3">${escapeHtml(latest.data.error || 'Deep dive metrics unavailable.')}</p>`;
				}
				const tables = await requestJson('/api/deepdive/tables?limit=10');
				let tablesContent = '';
				if (tables.ok && tables.data.status === 'ok') {
					tablesContent = renderDetailTables(tables.data.tables);
				} else if (tables.ok && tables.data.status === 'error') {
					tablesContent = `<p class="text-danger mb-0">${escapeHtml(tables.data.error || 'Unable to load detail tables.')}</p>`;
				} else if (!tables.ok) {
					tablesContent = `<p class="text-danger mb-0">${escapeHtml(tables.error || 'Failed to load detail tables.')}</p>`;
				}
				deepDiveResultDiv.innerHTML = html + tablesContent;
			}

			async function handleTraining() {
				if (!state.currentModel) {
					return;
				}
				const label = trainBtn.textContent;
				trainBtn.disabled = true;
				trainBtn.textContent = 'Training...';
				artifactsDiv.innerHTML = '<p class="text-muted mb-0">Executing notebook cells...</p>';
				setText(trainLogsDiv, 'Awaiting notebook logs...');
				const { ok, data, error } = await requestJson(`/api/train/${state.currentModel}`, { method: 'POST' });
				if (ok && data.status === 'ok') {
					if (data.run) {
						renderArtifacts(data.run);
					} else {
						await refreshArtifacts();
					}
					renderLogs(data.logs || []);
				} else if (ok) {
					artifactsDiv.innerHTML = `<p class="text-danger mb-0">${escapeHtml(data.error || 'Training failed.')}</p>`;
					renderErrors(data.details || []);
				} else {
					artifactsDiv.innerHTML = `<p class="text-danger mb-0">${escapeHtml(error || 'Training request failed.')}</p>`;
					renderErrors([]);
				}
				trainBtn.disabled = false;
				trainBtn.textContent = label;
			}

			themeToggle.addEventListener('click', () => {
				document.body.classList.toggle('dark-mode');
			});

			toggleSection('dataToggle', 'dataPreview', 'Hide Data', 'Show Data');
			toggleSection('healthToggle', 'healthResult', 'Hide Health', 'Show Health');
			toggleSection('deepDiveToggle', 'deepDiveResult', 'Hide Deep Dive', 'Show Deep Dive');
			toggleSection('artifactsToggle', 'artifacts', 'Hide Results', 'Show Results');

			modelSelect.addEventListener('change', async () => {
				state.currentModel = modelSelect.value;
				await refreshArtifacts();
			});

			showDataBtn.addEventListener('click', handleDataPreview);
			healthBtn.addEventListener('click', handleHealth);
			deepDiveBtn.addEventListener('click', handleDeepDive);
			trainBtn.addEventListener('click', handleTraining);

			setText(trainLogsDiv, 'No logs yet.');
			loadModels();
		});
		