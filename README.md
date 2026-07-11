# Electronics Supply Chain Disruption Predictor

Local SQLite, ChromaDB, RAG search, forecasting, and disruption-scenario
dashboard built around Varun's electronics/semiconductor workbook.

Yogita's beauty/FMCG dataset is not loaded into either database.

## Current data

Source workbook:

```text
data/raw/supply_chain_lite_master.xlsx
```

Additional static RAG context:

```text
data/raw/RAG_data/*.pdf
data/raw/RAG_data/*.docx
config/playbooks/*.txt
```

The database build preserves:

- 5,459 Lite Master order records
- 200 operational KPI records
- 2,282 semiconductor signal records
- Workbook data dictionary and legend
- Duplicate business order IDs without dropping rows

Generated outputs:

```text
outputs/supply_chain.db
outputs/chromadb/
```

## Requirements

- Python 3.11 or 3.12
- Internet access during the first setup to download Python packages and the
  `all-MiniLM-L6-v2` embedding model
- Internet access when running scenarios because weather data comes from
  Open-Meteo

No OpenAI API key is required.

## Setup on Windows PowerShell

Run all commands from the project root:

```powershell
cd D:\supply-chain-disrupter
```

Create and activate a virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation for the current session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Build SQLite and ChromaDB

```powershell
python -m src.build_databases
```

Expected headline results:

```text
SQLite: loaded 5,459 Lite Master orders
ChromaDB: 306 chunks
```

The command safely rebuilds:

- SQLite tables for Lite Master, operational KPIs, semiconductor signals,
  workbook metadata, and mitigation outputs
- A `daily_records` compatibility view used by the scenario workflow
- One electronics-only ChromaDB collection containing semiconductor events,
  mitigation knowledge, playbooks, event profiles, field definitions, and
  the committed PDF/DOCX static context

## Live data ingestion (Data Ingestion agent)

The Data Ingestion agent fetches live signals from external APIs and stores them
in SQLite, separate from the historical workbook tables:

- **Weather** — Open-Meteo, one row per configured hub per day (`weather_signals`).
- **News** — GDELT DOC 2.0 API with an RSS fallback, deduplicated (`news_signals`).

Both tables are stamped with `source_type` and `ingestion_ts` for provenance.

```powershell
python -m src.agents.data_ingestion.cli            # weather + news
python -m src.agents.data_ingestion.cli --weather  # weather only
python -m src.agents.data_ingestion.cli --news --rss   # news from GDELT + RSS
```

If GDELT requests fail with a TLS/certificate error on your network (common
behind corporate proxies), opt out of verification explicitly for the run:

```powershell
$env:INGEST_INSECURE_SSL = "1"
python -m src.agents.data_ingestion.cli --news
```

TLS verification stays on by default; the RSS fallback still returns news even
when GDELT is blocked, so the news table is never left empty.

## Run the application (Streamlit — original dashboard)

```powershell
python -m streamlit run src/main.py
```

Streamlit normally opens:

```text
http://localhost:8501
```

The application contains three pages:

1. **Data Ingestion** — rebuild and inspect SQLite and ChromaDB.
2. **RAG Search** — search semiconductor events, mitigations, and field
   definitions.
3. **Scenario Analyzer** — select an existing workbook region, product, and
   date; calculate risk, run a Prophet forecast, estimate stockout exposure,
   and persist mitigation guidance.

## Run the application (React dashboard — FastAPI + Vite)

The command center UI is a React app (`src/frontend/`) backed by a FastAPI
server (`src/api/main.py`). Both must be running at the same time, on fixed
ports, or the UI shows "couldn't load data":

- Backend: `http://127.0.0.1:8173`
- Frontend: `http://127.0.0.1:5173`

The Vite dev server proxies every `/api/*` request straight to the backend
(see `src/frontend/vite.config.ts`), so the frontend needs **no `.env` file**
to find the API — it works the same on every machine out of the box.

**One-time setup** (creates the venv, installs Python deps, installs npm
deps):

```powershell
.\scripts\setup.ps1
```

**Every time you want to run it**, open two terminals from the project root:

```powershell
# Terminal 1 — backend (must stay on port 8173)
.\scripts\run_backend.ps1

# Terminal 2 — frontend (must stay on port 5173)
.\scripts\run_frontend.ps1
```

Then open `http://localhost:5173`. If a port is already in use, the scripts
will fail loudly instead of silently switching ports (a common cause of
"can't connect to backend") — free the port or stop the other process first.

Prefer to run the commands directly instead of the scripts?

```powershell
# Backend — run from the project root so relative paths (outputs/, data/, config/) resolve
.\.venv\Scripts\Activate.ps1
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8173 --reload

# Frontend
cd src/frontend
npm install   # first time only
npm run dev
```

## Typical workflow (Streamlit path)

```powershell
cd D:\supply-chain-disrupter
.\.venv\Scripts\Activate.ps1
python -m src.build_databases
python -m src.agents.data_ingestion.cli
python -m streamlit run src/main.py
```

The databases only need to be rebuilt when the workbook, playbooks, or database
code changes.

## Project structure

```text
config/
  india_electronics.yaml       Port coordinates and fallback routes
  playbooks/                   Electronics disruption playbooks
data/
  raw/
    supply_chain_lite_master.xlsx
outputs/                       Generated SQLite and ChromaDB files
src/
  agents/
    data_ingestion/            L1: scenario loader + live weather/news ingestion
      agent.py                 data_ingestion_agent
      live_ingest.py           Open-Meteo + GDELT/RSS -> SQLite
      cli.py                   Ingestion CLI entry point
    weather_agent/             L3: live weather risk
      agent.py                 weather_risk_monitoring_agent
      client.py                Open-Meteo client + severity
    news_agent/                L2: news/event risk signals
      agent.py                 news_event_analysis_agent
      rag.py                   RAG signal builder
    state.py                   Shared Pydantic state models
    langgraph_engine.py        Orchestrator + L4-L7 agents
  dashboard/                   Streamlit pages
  utils/                       ETL, SQLite, RAG, and YAML utilities
  build_databases.py           Database build command
  main.py                      Streamlit entry point
requirements.txt
README.md
```

## Troubleshooting

### `No module named streamlit`

Activate the virtual environment and reinstall dependencies:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Database or collection is missing

```powershell
python -m src.build_databases
```

### Embedding model download warning

The first ChromaDB build downloads `all-MiniLM-L6-v2` from Hugging Face. A
Hugging Face token is optional; the model can be downloaded anonymously.

### React UI says "couldn't load data" / can't connect to backend

This means the FastAPI backend isn't running on port 8173, or something else
is already bound to port 5173 or 8173. Check:

```powershell
# Is the backend actually up?
curl http://127.0.0.1:8173/api/health

# Anything already using the ports?
netstat -ano | findstr ":8173"
netstat -ano | findstr ":5173"
```

Stop whatever is holding the port (or restart with `.\scripts\run_backend.ps1`
/ `.\scripts\run_frontend.ps1`, which fail fast instead of silently switching
ports). The frontend does not need a `.env.local` file — it talks to the
backend through the Vite proxy, not a configured URL.

### Scenario weather request fails

The database and RAG pages still work offline after initial setup. Scenario
weather enrichment requires access to:

```text
https://api.open-meteo.com
```
