"use client";

/**
 * Custom frameless title bar for the Electron window.
 * Rendered in the browser — only calls Electron IPC if window.peris exists
 * (i.e. when running inside Electron). In plain browser dev mode it still
 * renders but the buttons are no-ops so the UI looks consistent.
 */

declare global {
  interface Window {
    peris?: {
      minimizeWindow: () => void;
      maximizeWindow: () => void;
      closeWindow: () => void;
      getAppVersion: () => Promise<string>;
      onStatusUpdate: (cb: (data: { step: number; message: string }) => void) => void;
    };
  }
}

export default function TitleBar() {
  const minimize = () => window.peris?.minimizeWindow();
  const maximize = () => window.peris?.maximizeWindow();
  const close    = () => window.peris?.closeWindow();

  // Only render in Electron (when window.peris is present)
  // In browser dev mode we skip it so the layout stays normal
  if (typeof window !== "undefined" && !window.peris) return null;

  return (
    <div
      className="flex items-center justify-between bg-gray-900 border-b border-gray-800 select-none"
      style={{ height: 36, WebkitAppRegion: "drag" } as React.CSSProperties}
    >
      {/* Left: app identity */}
      <div className="flex items-center gap-2 px-4">
        <div className="w-5 h-5 bg-blue-600 rounded flex items-center justify-center text-white font-bold text-xs">
          P
        </div>
        <span className="text-xs font-semibold text-gray-300 tracking-widest uppercase">
          PERIS
        </span>
      </div>

      {/* Right: window controls — no-drag zone */}
      <div
        className="flex items-center"
        style={{ WebkitAppRegion: "no-drag" } as React.CSSProperties}
      >
        <button
          onClick={minimize}
          title="Minimise"
          className="w-9 h-9 flex items-center justify-center text-gray-500 hover:text-gray-200 hover:bg-gray-700 transition-colors"
        >
          <svg width="10" height="1" viewBox="0 0 10 1" fill="currentColor">
            <rect width="10" height="1" />
          </svg>
        </button>
        <button
          onClick={maximize}
          title="Maximise"
          className="w-9 h-9 flex items-center justify-center text-gray-500 hover:text-gray-200 hover:bg-gray-700 transition-colors"
        >
          <svg width="10" height="10" viewBox="0 0 10 10" fill="none" stroke="currentColor" strokeWidth="1">
            <rect x="0.5" y="0.5" width="9" height="9" />
          </svg>
        </button>
        <button
          onClick={close}
          title="Close"
          className="w-9 h-9 flex items-center justify-center text-gray-500 hover:text-white hover:bg-red-600 transition-colors"
        >
          <svg width="10" height="10" viewBox="0 0 10 10" stroke="currentColor" strokeWidth="1.2">
            <line x1="0" y1="0" x2="10" y2="10" />
            <line x1="10" y1="0" x2="0" y2="10" />
          </svg>
        </button>
      </div>
    </div>
  );
}
