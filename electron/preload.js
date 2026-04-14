'use strict';

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('peris', {
  // Called by splash screen to receive step-by-step status updates
  onStatusUpdate: (callback) => {
    ipcRenderer.on('status-update', (_event, data) => callback(data));
  },

  // App metadata
  getAppVersion: () => ipcRenderer.invoke('get-app-version'),

  // Custom window controls (used by the frameless title bar)
  minimizeWindow: () => ipcRenderer.send('window-minimize'),
  maximizeWindow: () => ipcRenderer.send('window-maximize'),
  closeWindow:    () => ipcRenderer.send('window-close'),
});
