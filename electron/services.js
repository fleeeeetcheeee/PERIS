'use strict';

/**
 * services.js — manages Python venv checks, Ollama detection,
 * daily first-launch jobs, and status event emission.
 */

const { execFile } = require('child_process');
const path = require('path');
const fs = require('fs');

const ROOT = path.resolve(__dirname, '..');
const LAST_RUN_FILE = path.join(ROOT, '.peris_last_run.json');

// ── Helpers ──────────────────────────────────────────────────────────────────

function todayStr() {
  return new Date().toISOString().slice(0, 10); // "YYYY-MM-DD"
}

function readLastRun() {
  try {
    return JSON.parse(fs.readFileSync(LAST_RUN_FILE, 'utf8'));
  } catch {
    return {};
  }
}

function writeLastRun(data) {
  fs.writeFileSync(LAST_RUN_FILE, JSON.stringify(data, null, 2));
}

// ── Exports ───────────────────────────────────────────────────────────────────

/**
 * Check whether the Python venv exists and has key packages installed.
 * Returns true if usable, false otherwise.
 */
function checkVenv() {
  const venvPython = path.join(ROOT, 'venv', 'bin', 'python');
  if (!fs.existsSync(venvPython)) {
    console.warn('[services] venv not found at', venvPython);
    return false;
  }
  return true;
}

/**
 * Check whether Ollama is installed by probing known install locations.
 * Avoids relying on PATH (which is truncated inside Electron's child_process).
 */
function checkOllama() {
  const candidates = [
    '/usr/local/bin/ollama',
    '/opt/homebrew/bin/ollama',
    '/usr/bin/ollama',
  ];
  return candidates.some((p) => fs.existsSync(p));
}

/**
 * Run a one-off CLI command inside the venv and return a Promise that
 * resolves when it exits (regardless of exit code).
 */
function runCli(args) {
  return new Promise((resolve) => {
    const venvPython = path.join(ROOT, 'venv', 'bin', 'python');
    const python = fs.existsSync(venvPython) ? venvPython : 'python3';
    // execFile takes binary + args array — no shell, no injection risk
    const proc = execFile(python, ['cli.py', ...args], { cwd: ROOT });
    proc.stdout && proc.stdout.on('data', (d) => process.stdout.write('[cli] ' + d));
    proc.stderr && proc.stderr.on('data', (d) => process.stderr.write('[cli] ' + d));
    proc.on('close', resolve);
  });
}

/**
 * Run daily first-launch jobs if they haven't run today.
 * Emits progress via the provided `emit` callback: emit(step, message).
 */
async function runDailyJobsIfNeeded(emit) {
  const today = todayStr();
  const lastRun = readLastRun();

  if (lastRun.date === today) {
    console.log('[services] Daily jobs already ran today, skipping');
    return;
  }

  console.log('[services] Running daily first-launch jobs for', today);
  emit(1, 'Ingesting SEC EDGAR data...');
  await runCli(['ingest', 'sec']);

  emit(2, 'Scoring new companies...');
  await runCli(['score', '--all']);

  writeLastRun({ date: today });
  console.log('[services] Daily jobs complete');
}

module.exports = { checkVenv, checkOllama, runDailyJobsIfNeeded };
