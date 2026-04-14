'use strict';

/**
 * electron-builder configuration for PERIS desktop app.
 * Run: npm run build  (builds Next.js static export, then packages with electron-builder)
 */

const path = require('path');

/** @type {import('electron-builder').Configuration} */
module.exports = {
  appId: 'com.peris.intelligence',
  productName: 'PERIS',
  copyright: 'Copyright © 2026 PERIS',

  // Source of the Electron main process (inside electron/ folder)
  directories: {
    app: 'electron',
    output: 'dist',
    buildResources: 'electron/build-resources',
  },

  // Files to bundle into the app package
  files: [
    'main.js',
    'preload.js',
    'services.js',
    'splash.html',
    'package.json',
    'node_modules/**/*',
  ],

  // Extra files copied next to the app (outside the asar archive for Python access)
  extraResources: [
    { from: '../src',          to: 'src',          filter: ['**/*'] },
    { from: '../venv',         to: 'venv',         filter: ['**/*'] },
    { from: '../frontend/out', to: 'frontend/out', filter: ['**/*'] },
    { from: '../scheduler.py', to: 'scheduler.py' },
    { from: '../cli.py',       to: 'cli.py' },
    { from: '../thesis.txt',   to: 'thesis.txt',   filter: ['**/*'] },
    { from: '../peris.db',     to: 'peris.db' },
  ],

  // macOS
  mac: {
    category: 'public.app-category.finance',
    target: [
      { target: 'dmg', arch: ['universal'] },
    ],
    icon: 'electron/icon.png',
    hardenedRuntime: true,
    gatekeeperAssess: false,
  },

  dmg: {
    title: 'PERIS Installer',
    window: { width: 540, height: 380 },
    contents: [
      { x: 130, y: 220, type: 'file' },
      { x: 410, y: 220, type: 'link', path: '/Applications' },
    ],
  },

  // Linux (optional)
  linux: {
    target: ['AppImage'],
    category: 'Finance',
  },

  // Windows (optional)
  win: {
    target: ['nsis'],
  },

  // Hooks
  afterPack: async (context) => {
    // Placeholder for any post-pack operations (e.g., code signing prep)
    console.log('afterPack:', context.appOutDir);
  },
};
