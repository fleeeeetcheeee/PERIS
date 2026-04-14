# PERIS Electron Desktop App

Wraps the PERIS FastAPI backend + Next.js frontend in a native Electron window.

---

## Prerequisites

- Node.js ≥ 18
- Python 3.11 with the PERIS venv at `../venv/`
- Ollama installed (optional — used for local LLM scoring)

---

## Dev mode

```bash
# From the repo root (PERIS/)
npm install          # install root devDependencies
npm run dev          # starts Next.js + Electron concurrently
```

What happens:
1. `npm run dev:frontend` — starts Next.js on port 3000
2. `npm run dev:electron` — waits for port 3000 then opens the Electron window
3. Electron spawns the FastAPI backend (uvicorn on port 8000) and Ollama (if installed)
4. Splash screen shows service startup progress
5. Main window loads `http://localhost:3000` once both services are ready

---

## Production build

```bash
# 1. Build the Next.js static export
npm --prefix frontend run build

# 2. Package with electron-builder (produces dist/PERIS-*.dmg)
npm --prefix electron run build
```

The packaged app embeds:
- `src/` — Python API source
- `venv/` — Python virtual environment (all dependencies)
- `frontend/out/` — Next.js static export
- `scheduler.py`, `cli.py`, `thesis.txt`, `peris.db`

---

## Distributing the .dmg

After `npm run build`:

```
dist/
  PERIS-0.1.0-universal.dmg   ← drag to /Applications
```

Mount the DMG, drag PERIS to Applications, launch. The first run will:
1. Start the embedded FastAPI backend
2. Start Ollama (if installed system-wide)
3. Run daily SEC EDGAR ingest + scoring (if not already run today)

---

## Customising the icon

Replace `electron/icon.png` with a 512×512 PNG.  
For macOS, electron-builder will auto-convert to `.icns`.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Splash hangs on "Starting API server" | Check uvicorn is in `venv/bin/` and `peris.db` is accessible |
| Blank white window | Next.js dev server hasn't started yet — wait ~15 s |
| "venv not found" in console | Run `python3 -m venv venv && venv/bin/pip install -r requirements.txt` from repo root |
| Ollama not starting | Install Ollama from https://ollama.ai and ensure `ollama` is on PATH |
