'use strict';

/** @type {import('electron-builder').Configuration} */
module.exports = {
  appId: 'com.peris.intelligence',
  productName: 'PERIS',
  copyright: 'Copyright © 2026 PERIS',

  electronVersion: '41.2.0',

  directories: {
    output: 'dist',
    buildResources: 'assets',
  },

  files: [
    'electron/**/*',
    'frontend/.next/**/*',
    'frontend/public/**/*',
    'src/**/*',
    'scheduler.py',
    'cli.py',
    'thesis.txt',
    '*.py',
    'requirements.txt',
  ],

  mac: {
    category: 'public.app-category.finance',
    target: 'dmg',
    icon: 'assets/icon.icns',
    identity: null,
  },

  dmg: {
    title: 'PERIS Installer',
  },
};
