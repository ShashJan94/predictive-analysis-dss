# WSB Decision Support System

A full-stack decision support application that combines automated data health monitoring, predictive modeling, and NLP sentiment analysis through a unified web dashboard. Built with FastAPI and DuckDB, WSB-DSS orchestrates Jupyter notebook workflows to deliver reproducible analytics and model training pipelines for Airbnb Seattle datasets.

## Overview

WSB-DSS provides an interactive web interface for:
- **Data Quality Monitoring** – Automated health checks and deep-dive diagnostics
- **Predictive Modeling** – Regression, classification, clustering, and time-series forecasting
- **NLP Analytics** – Multilingual sentiment analysis on customer reviews
- **Artifact Management** – Centralized storage and retrieval of models, metrics, and visualizations

All analytics workflows are powered by `WSB-DSS.ipynb`, with the FastAPI backend (`app.py`) orchestrating selective cell execution to ensure consistency and reproducibility.

---

## Project Structure

```
WSB-DSS/
├── app.py                      # FastAPI application with embedded dashboard
├── WSB-DSS.ipynb               # Core analytics notebook (orchestrated by API)
├── requirements.txt            # Python dependencies
├── wsb_dss.duckdb              # DuckDB database (audit results, run history)
├── README.md                   # This file
├── artifacts/                  # Model outputs, metrics, plots
│   ├── regression/
│   ├── logistic/
│   ├── kmeans/
│   ├── forecast/
│   └── nlp/
├── data/
│   └── airbnb_seattle/         # Input datasets
│       ├── calendar.csv
│       ├── listings.csv
│       └── reviews.csv
└── kaggle_key/
    └── kaggle.json             # Kaggle API credentials (optional, gitignored)
```

---

## Prerequisites

- **Python**: 3.10 or later
- **Conda**: Recommended for environment management
- **Kaggle API Key**: Optional (only if downloading datasets from Kaggle)

---

## Installation & Setup

### 1. Clone the Repository

```powershell
git clone <repository-url>
cd WSB-DSS
```

### 2. Create Conda Environment

```powershell
# Create new environment with Python 3.10
conda create -n wsb-dss python=3.10 -y

# Activate the environment
conda activate wsb-dss
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

The `requirements.txt` includes:
- **FastAPI** – Web framework and API server
- **Uvicorn** – ASGI server for FastAPI
- **DuckDB** – Embedded analytics database
- **pandas**, **numpy** – Data manipulation
- **scikit-learn** – Machine learning algorithms
- **matplotlib**, **seaborn** – Visualization
- **transformers**, **torch** – NLP sentiment analysis
- **nbformat**, **nbclient** – Notebook execution engine
- **kaggle** – Dataset download (optional)

### 4. Configure Kaggle Credentials (Optional)

If you need to download datasets from Kaggle:

1. Obtain your Kaggle API token from [kaggle.com/account](https://www.kaggle.com/account)
2. Save `kaggle.json` in the `kaggle_key/` directory
3. Set the environment variable:

```powershell
# PowerShell
$env:KAGGLE_CONFIG_DIR = "$(Get-Location)\kaggle_key"
```

The notebook's Kaggle download cells will auto-detect this configuration.

### 5. Verify Data Files

Ensure the following files exist in `data/airbnb_seattle/`:
- `calendar.csv`
- `listings.csv`
- `reviews.csv`

If missing, either:
- Download manually from [Kaggle Airbnb Seattle dataset](https://www.kaggle.com/airbnb/seattle)
- Run the Kaggle download cells in `WSB-DSS.ipynb` (requires step 4)

---

## Running the Application

### Start the FastAPI Server

```powershell
# From the project root directory
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

**Options:**
- `--host 0.0.0.0` – Accept connections from any network interface
- `--port 8000` – Listen on port 8000
- `--reload` – Auto-restart on code changes (development mode)

### Access the Dashboard

Open your browser and navigate to:
- **Dashboard**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **API Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) (Swagger UI)

---

## Using the Dashboard

### 1. Dataset Preview
- Click **Show Data** to load a preview of the source CSV files
- Toggle visibility with **Hide Data**
- Displays the first 5 rows of each dataset

### 2. Data Health Checks
- Click **Check Health** to run automated quality audits:
  - Row/column counts
  - Missing value analysis
  - Duplicate detection
  - Referential integrity checks
  - Date range validation
- Results are stored in DuckDB and displayed in expandable tables

### 3. Deep Dive Analysis
- Click **Deep Dive** for advanced diagnostics:
  - Occupancy rates by listing
  - Price missingness patterns
  - Review activity gaps
  - Neighborhood-level summaries
- Includes both summary metrics and detailed data tables

### 4. Model Training
1. **Select a model** from the dropdown:
   - **Regression (HGB)** – Price prediction using HistGradientBoosting
   - **Logistic Regression** – Price bucket classification
   - **KMeans Clustering** – Listing segmentation
   - **Calendar Forecast** – Time-series demand prediction
   - **NLP Sentiment Analysis** – Review sentiment scoring

2. Click **Train** to execute the model pipeline
3. Monitor execution logs in the **Notebook Outputs** section
4. View results in the **Model Outputs** panel:
   - Performance metrics
   - Visualizations (plots, charts)
   - Downloadable artifacts (models, CSVs)

### 5. Artifacts & Results
- All model outputs are saved under `artifacts/<model_type>/<run_id>/`
- The dashboard automatically displays the latest run
- Click artifact links to download models, metrics JSON, or output files

---

## Architecture & Workflow

### Notebook-Driven Execution
- The FastAPI backend executes specific cells from `WSB-DSS.ipynb` based on the requested operation
- Cell ranges are pre-defined in `app.py` (`MODEL_TRAIN_CELLS` dictionary)
- Execution is serialized via threading locks to prevent conflicts

### Cell Mapping Example
```python
MODEL_TRAIN_CELLS = {
    "regression": [1, 4, 5, 6, 11, 15, 16],  # Import → Load → Train → Persist
    "logistic":   [1, 4, 5, 6, 11, 12, 15, 17],
    "kmeans":     [1, 4, 5, 6, 11, 12, 15, 18],
    "forecast":   [1, 4, 5, 6, 13, 15, 19],
    "nlp":        [1, 4, 5, 6, 14, 15, 20],
}
```

### Data Flow
1. User triggers action via dashboard
2. FastAPI endpoint invokes `execute_notebook_cells()`
3. Selected cells run in an isolated Jupyter kernel
4. Outputs are captured and returned as JSON
5. Artifacts are persisted to disk and logged in DuckDB
6. Dashboard renders results and provides download links

---

## API Endpoints

### Core Endpoints
- `GET /` – Serve web dashboard
- `GET /api/models` – List available models
- `POST /api/train/{model_id}` – Train a specific model
- `GET /api/artifacts/{model_id}` – Retrieve latest artifacts
- `GET /api/data/sample` – Preview dataset rows

### Health & Diagnostics
- `GET /api/health/latest` – Fetch cached health metrics
- `POST /api/health/run` – Execute health audit notebook
- `GET /api/deepdive/latest` – Fetch cached deep-dive results
- `GET /api/deepdive/tables` – Retrieve detail tables

### Artifact Files
- `GET /artifacts/file/{model_id}/{run_id}/{filename}` – Download specific artifact

Full API documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Troubleshooting

### Dashboard Not Loading
- **Check server logs** for Python exceptions
- **Verify port 8000** is not in use: `netstat -ano | findstr :8000`
- **Clear browser cache** and reload

### JavaScript Errors
- Open browser DevTools (F12) → Console tab
- Common issues: Syntax errors, failed fetch requests
- Ensure the server is running and accessible

### Notebook Execution Failures
- **Review terminal output** for cell-level errors
- **Verify data files** exist in `data/airbnb_seattle/`
- **Check cell indices** in `app.py` match the notebook structure
- **Validate dependencies** are installed: `pip list`

### Missing Artifacts
- Ensure the `artifacts/` directory has write permissions
- Check that the model run completed without errors
- Inspect DuckDB for run history: `SELECT * FROM runs;`

### DuckDB Lock Errors
- Only one operation can write to DuckDB at a time
- Wait for ongoing tasks to complete before triggering new ones
- Restart the server if locks persist

### Kaggle Download Issues
- Verify `kaggle.json` exists in `kaggle_key/`
- Set `KAGGLE_CONFIG_DIR` environment variable
- Check Kaggle API quota and credentials

---

## Development Notes

### Modifying the Notebook
1. Edit `WSB-DSS.ipynb` to update analytics logic
2. Update cell indices in `app.py` if structure changes
3. Test manually in Jupyter before deploying
4. Restart uvicorn to pick up notebook changes

### Adding New Models
1. Create training function in a new notebook cell
2. Add persistence logic in the artifacts helper cell
3. Register the model in `MODELS` list in `app.py`
4. Define cell execution range in `MODEL_TRAIN_CELLS`
5. Test via API endpoint: `POST /api/train/{new_model_id}`

### Database Schema
The DuckDB database (`wsb_dss.duckdb`) stores:
- **health_checks** – Audit metrics (JSON)
- **deep_dive_checks** – Advanced diagnostics (JSON)
- **runs** – Model execution history
- **detail_deepdive_*** – Normalized detail tables

Query examples:
```sql
-- View latest health check
SELECT * FROM health_checks ORDER BY computed_at DESC LIMIT 1;

-- List all model runs
SELECT * FROM runs WHERE model_type = 'regression' ORDER BY end_time DESC;
```

---

## Future Enhancements

- [ ] Add user authentication for production deployment
- [ ] Implement automated test suite (pytest)
- [ ] Containerize with Docker for portable deployment
- [ ] Add real-time logging stream to dashboard
- [ ] Support concurrent notebook execution with job queuing
- [ ] Integrate CI/CD pipeline for automated testing

---

## License

This project is developed for academic purposes as part of the WSB University Decision Support Systems course.

---

## Contact & Support

For questions, issues, or contributions, please open an issue in the repository or contact the development team.
