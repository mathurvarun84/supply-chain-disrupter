# Agentic Supply Chain Disruption Predictor & Simulation Engine

## Overview and Problem Statement

This project aims to develop an AI-powered supply chain disruption prediction and simulation system that proactively monitors global risk signals, predicts potential supply chain disruptions, forecasts demand impact, and simulates mitigation scenarios using multi-agent AI workflows, Retrieval-Augmented Generation (RAG), time-series forecasting, and discrete-event simulation techniques.

Modern supply chains are highly interconnected networks involving suppliers, manufacturers, logistics providers, ports, warehouses, and retailers across multiple countries. Disruptions such as extreme weather events, geopolitical conflicts, labor strikes, port closures, tariffs, or factory shutdowns can significantly affect inventory availability, delivery timelines, and revenue.

Most organizations react only after disruptions occur because monitoring large-scale real-time data sources manually is difficult and inefficient. Businesses require intelligent systems capable of continuously analyzing news feeds, weather alerts, shipping indices, and logistics signals to identify risks before they escalate.

The proposed system will build a LangGraph-orchestrated multi-agent platform that continuously ingests disruption-related information, classifies supply chain risks, forecasts demand fluctuations, simulates disruption scenarios, and generates mitigation recommendations. The platform aims to support proactive risk management, improve operational resilience, and assist organizations in strategic supply chain planning.

## Methodology

The platform is orchestrated using a LangGraph-based multi-agent workflow, where each agent is responsible for a specialized task:

1. **Real-Time Data Ingestion Agent:**
   - Continuously collects data from RSS feeds, logistics APIs, weather APIs, and shipping indices.
   - Extracts disruption-related events using `feedparser` and NLP pipelines.
   - Stores incoming disruption signals in a structured format.

2. **News and Event Analysis Agent:**
   - Analyzes news articles and logistics alerts using LLMs.
   - Identifies disruption categories such as weather, geopolitical, logistics, raw material shortages, and demand shocks.

3. **Weather Risk Monitoring Agent:**
   - Fetches weather forecasts using the Open-Meteo API.
   - Detects extreme weather events affecting ports, transportation hubs, and manufacturing facilities.

4. **Risk Classification Agent:**
   - Built within a LangGraph-based orchestration workflow.
   - Uses DistilBERT-based classifiers to categorize supply chain risks.
   - Generates supplier-level and trade-lane risk scores.

5. **Demand Forecasting Agent:**
   - Trains forecasting models using Facebook Prophet.
   - Incorporates disruption risk signals into demand forecasting pipelines.
   - Predicts demand deviations and inventory fluctuations.

6. **Simulation Agent:**
   - Simulates supply chain disruptions using SimPy discrete-event simulation.
   - Models suppliers, warehouses, ports, and retailers as network nodes.
   - Runs Monte Carlo simulations to estimate stockout probability and revenue impact.

7. **Mitigation Recommendation Agent:**
   - Generates natural-language mitigation recommendations using LLMs.
   - Suggests alternate suppliers, route changes, safety stock adjustments, and inventory actions.

8. **Dashboard and Alerting:**
   - A Gradio dashboard for risk visualization and scenario analysis.
   - Displays disruption heatmaps, timelines, and simulation outcomes.
   - Triggers alerts for high-risk scenarios.

## Datasets

The project will utilize a combination of public datasets, APIs, and synthetic disruption datasets:
- **SupplyChainNet Dataset:** Historical supply chain transactions, shipping records, supplier information, logistics delays, and disruption events (Source: Kaggle).
- **Freightos Baltic Index Data:** Container shipping rate indices, freight trends, and logistics cost fluctuations (Source: Public Freightos shipping index).
- **Open-Meteo Weather API:** Historical and forecast weather data for logistics hubs and shipping regions.
- **News and RSS Feed Data:** News articles related to labor strikes, geopolitical risks, tariffs, factory shutdowns, and logistics disruptions (Sources: Reuters, Bloomberg, Supply Chain Dive, Google News API).
- **Synthetic Supply Chain Events:** AI-generated synthetic disruption events, simulated demand shocks, disruption narratives, supplier failures, and logistics incidents.

## Project Structure

```text
.
├── config/                 # Configuration files (API keys, model parameters)
├── data/
│   ├── processed/          # Cleaned and structured data
│   └── raw/                # Raw datasets from Kaggle, synthetic data, etc.
├── notebooks/              # Jupyter notebooks for EDA and prototyping
├── src/                    # Main source code
│   ├── agents/             # LangGraph agent implementations
│   ├── dashboard/          # Gradio UI components
│   ├── models/             # ML models (DistilBERT, Prophet wrappers)
│   ├── simulation/         # SimPy discrete-event simulation logic
│   └── utils/              # Helper functions, API clients, data parsers
├── tests/                  # Unit and integration tests
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## Evaluation

- Measure risk classification accuracy.
- Evaluate demand forecast deviation.
- Assess simulation realism and mitigation recommendation quality.

## Challenges

- **Real-Time Data Integration:** Continuously processing multiple external data sources reliably.
- **Risk Signal Extraction:** Identifying meaningful disruption indicators from noisy news and weather data.
- **Forecasting Uncertainty:** Handling uncertainty in demand forecasting under disruption conditions.
- **Simulation Complexity:** Modeling realistic supply chain behavior and interconnected dependencies.
- **Multi-Agent Coordination:** Synchronizing ingestion, forecasting, simulation, and mitigation agents effectively.
- **Scalability:** Efficiently managing large-scale supplier networks and logistics data.
- **Recommendation Reliability:** Generating actionable and business-relevant mitigation strategies.

## Technologies Used

- **Agent Orchestration:** LangGraph, LangChain
- **Machine Learning & NLP:** Hugging Face Transformers (DistilBERT), PyTorch
- **Time-Series Forecasting:** Facebook Prophet
- **Simulation:** SimPy (Discrete-Event Simulation)
- **Data Collection:** Feedparser, Requests
- **UI & Dashboard:** Gradio
- **Data Processing:** Pandas, NumPy

## References

- [Facebook Prophet Documentation](https://facebook.github.io/prophet/)
- [SimPy Documentation](https://simpy.readthedocs.io/en/latest/)
- [Feedparser Documentation](https://feedparser.readthedocs.io/en/latest/)
- [DistilBERT Model Documentation](https://huggingface.co/distilbert-base-uncased)
