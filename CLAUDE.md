# PERIS — Private Equity Research Intelligence System

## Project purpose
Locally-hosted PE deal intelligence tool. No paid data sources.
Full pipeline: sourcing → thesis validation → pipeline CRM → portfolio monitoring → reporting.

## Stack
- Backend: Python 3.11, FastAPI, APScheduler, LangChain, Playwright
- Database: SQLite (structured), ChromaDB (vector/embeddings)
- LLM: Ollama (local), Claude API (synthesis/reports)
- Frontend: Next.js 14, Tailwind CSS, Recharts
- Scraping: Playwright, httpx, feedparser

## Folder structure
peris/
├── CLAUDE.md
├── README.md
├── .env.example
├── requirements.txt
├── package.json
├── src/
│   ├── ingestion/          # Data collectors, one file per source
│   ├── scoring/            # LLM-based screening and ranking
│   ├── pipeline/           # CRM logic, stage management
│   ├── monitoring/         # Portfolio alert agents
│   ├── reporting/          # PDF + Notion report generation
│   ├── integrations/       # API clients (SEC, FRED, Yahoo, etc.)
│   ├── db/                 # SQLite schema, migrations, queries
│   ├── agents/             # LangChain agent definitions
│   └── api/                # FastAPI routes
├── frontend/               # Next.js app
├── scheduler.py            # APScheduler entry point
├── cli.py                  # CLI for manual runs
└── tests/

## Data sources (all free)
- SEC EDGAR: https://efts.sec.gov/LATEST/search-index
- FRED: https://fred.stlouisfed.org/docs/api/fred/
- World Bank: https://datahelpdesk.worldbank.org/knowledgebase/topics/125589
- OpenCorporates: https://api.opencorporates.com
- USPTO Patents: https://developer.uspto.gov
- Alpha Vantage: https://www.alphavantage.co/documentation/
- Yahoo Finance: via yfinance Python lib
- Google Trends: via pytrends
- Reddit: via PRAW (free API)
- RSS feeds: Reuters, SEC 8-K alerts via feedparser

## Naming conventions
- Files: snake_case
- Classes: PascalCase
- All API clients live in src/integrations/ and inherit BaseIntegration
- All agents inherit BaseAgent from src/agents/base.py
- DB queries go in src/db/queries.py only — no raw SQL elsewhere

## Current sprint
1. Scaffold folder structure
2. SQLite schema (companies, pipeline_stages, portfolio_kpis, signals)
3. SEC EDGAR ingestion module
4. FRED macro data module
5. Company scoring agent (LangChain)
6. FastAPI routes for dashboard
7. Next.js frontend shell

## Agent roles
- Codex: scaffold boilerplate, API client stubs, test skeletons
- Cursor: implement logic inside stubs, data transforms, LangChain chains
- Claude Code: orchestration, integration tests, CLAUDE.md updates, CLI