'use strict';

const {
  app,
  BrowserWindow,
  ipcMain,
  screen,
} = require('electron');
const path = require('path');
const fs = require('fs');
const { execFile, spawn } = require('child_process');

const { checkVenv, checkOllama, runDailyJobsIfNeeded } = require('./services');

const ROOT       = path.resolve(__dirname, '..');
const VENV_BIN   = path.join(ROOT, 'venv', 'bin');
const PYTHON     = fs.existsSync(path.join(VENV_BIN, 'python'))
  ? path.join(VENV_BIN, 'python')
  : 'python3';

const BACKEND_PORT    = 8000;
const FRONTEND_PORT   = 3000;
const BACKEND_URL     = `http://localhost:${BACKEND_PORT}`;
const FRONTEND_URL    = `http://localhost:${FRONTEND_PORT}`;

// Track child processes for clean shutdown
const children = [];

// ── Window handles ────────────────────────────────────────────────────────────
let splashWin = null;
let mainWin   = null;

// ── Splash window ─────────────────────────────────────────────────────────────
function createSplash() {
  splashWin = new BrowserWindow({
    width: 440,
    height: 340,
    frame: false,
    resizable: false,
    center: true,
    transparent: false,
    backgroundColor: '#0f0f1a',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  splashWin.loadFile(path.join(__dirname, 'splash.html'));
}

function sendStatus(step, message) {
  if (splashWin && !splashWin.isDestroyed()) {
    splashWin.webContents.send('status-update', { step, message });
  }
  console.log(`[status ${step}] ${message}`);
}

// ── Main window ───────────────────────────────────────────────────────────────
function createMainWindow() {
  const { width, height } = screen.getPrimaryDisplay().workAreaSize;
  mainWin = new BrowserWindow({
    width: Math.min(1400, width),
    height: Math.min(900, height),
    frame: false,
    titleBarStyle: 'hidden',
    vibrancy: 'dark',
    backgroundColor: '#0f0f1a',
    show: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  mainWin.loadURL(FRONTEND_URL);

  mainWin.once('ready-to-show', () => {
    if (splashWin && !splashWin.isDestroyed()) splashWin.destroy();
    mainWin.show();
  });

  mainWin.on('closed', () => { mainWin = null; });
}

// ── IPC handlers ──────────────────────────────────────────────────────────────
ipcMain.handle('get-app-version', () => app.getVersion());
ipcMain.on('window-minimize', () => mainWin && mainWin.minimize());
ipcMain.on('window-maximize', () => {
  if (!mainWin) return;
  mainWin.isMaximized() ? mainWin.unmaximize() : mainWin.maximize();
});
ipcMain.on('window-close', () => app.quit());

// ── Process launchers ─────────────────────────────────────────────────────────

/** Spawn the FastAPI backend (src/api/main.py via uvicorn). */
function spawnBackend() {
  const uvicorn = path.join(VENV_BIN, 'uvicorn');
  const bin = fs.existsSync(uvicorn) ? uvicorn : 'uvicorn';
  const proc = spawn(bin, ['src.api.main:app', '--host', '127.0.0.1', '--port', String(BACKEND_PORT)], {
    cwd: ROOT,
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  proc.stdout.on('data', (d) => process.stdout.write('[api] ' + d));
  proc.stderr.on('data', (d) => process.stderr.write('[api] ' + d));
  proc.on('close', (code) => console.log('[api] exited with code', code));
  children.push(proc);
  return proc;
}

/** Spawn the Next.js dev server. */
function spawnFrontend() {
  const npmBin = process.platform === 'win32' ? 'npm.cmd' : 'npm';
  const proc = spawn(npmBin, ['run', 'dev'], {
    cwd: path.join(ROOT, 'frontend'),
    stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env, PORT: String(FRONTEND_PORT) },
  });
  proc.stdout.on('data', (d) => process.stdout.write('[next] ' + d));
  proc.stderr.on('data', (d) => process.stderr.write('[next] ' + d));
  proc.on('close', (code) => console.log('[next] exited with code', code));
  children.push(proc);
  return proc;
}

/** Spawn Ollama serve if ollama is installed and not already listening. */
function spawnOllamaIfNeeded() {
  if (!checkOllama()) {
    console.log('[ollama] not installed — skipping');
    return;
  }
  // Quick check: can we connect to ollama's default port (11434)?
  const net = require('net');
  const sock = net.createConnection({ port: 11434, host: '127.0.0.1' });
  sock.on('connect', () => { sock.destroy(); console.log('[ollama] already running'); });
  sock.on('error', () => {
    sock.destroy();
    console.log('[ollama] starting ollama serve...');
    const proc = spawn('ollama', ['serve'], { stdio: 'ignore', detached: false });
    proc.on('close', (code) => console.log('[ollama] exited with code', code));
    children.push(proc);
  });
}

/** Poll a URL until it responds 200 (or times out after `timeoutMs`). */
function waitForURL(url, timeoutMs = 60000) {
  return new Promise((resolve, reject) => {
    const start = Date.now();
    const http  = require('http');

    function attempt() {
      http.get(url, (res) => {
        res.resume(); // drain
        if (res.statusCode < 500) { resolve(); return; }
        retry();
      }).on('error', retry);
    }

    function retry() {
      if (Date.now() - start > timeoutMs) {
        reject(new Error(`Timed out waiting for ${url}`));
        return;
      }
      setTimeout(attempt, 800);
    }

    attempt();
  });
}

// ── App lifecycle ─────────────────────────────────────────────────────────────

app.whenReady().then(async () => {
  createSplash();

  // Give the splash a moment to paint
  await new Promise((r) => setTimeout(r, 400));

  sendStatus(0, 'Starting database...');

  // Check environment
  const venvOk = checkVenv();
  if (!venvOk) {
    console.warn('[main] venv missing — backend may fail to start');
  }

  sendStatus(1, 'Loading models...');
  spawnOllamaIfNeeded();

  sendStatus(2, 'Starting API server...');
  spawnBackend();
  spawnFrontend();

  // Wait for backend
  try {
    await waitForURL(BACKEND_URL + '/health', 90000);
    console.log('[main] backend ready');
  } catch (err) {
    console.warn('[main] backend health check timed out:', err.message);
  }

  sendStatus(3, 'Launching dashboard...');

  // Wait for Next.js
  try {
    await waitForURL(FRONTEND_URL, 90000);
    console.log('[main] frontend ready');
  } catch (err) {
    console.warn('[main] frontend health check timed out:', err.message);
  }

  sendStatus(4, 'Ready.');

  // Run daily ingest/score if first launch today (non-blocking)
  runDailyJobsIfNeeded((step, msg) => sendStatus(step, msg)).catch((e) =>
    console.warn('[main] daily jobs error:', e.message)
  );

  createMainWindow();
});

// Kill all spawned children on quit
app.on('will-quit', () => {
  for (const proc of children) {
    try { proc.kill(); } catch { /* already gone */ }
  }
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('activate', () => {
  if (mainWin === null) createMainWindow();
});
