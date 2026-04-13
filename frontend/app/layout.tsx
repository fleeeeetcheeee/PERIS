import type { Metadata } from "next";
import "./globals.css";
import Sidebar from "./components/Sidebar";
import Providers from "./lib/providers";

export const metadata: Metadata = {
  title: "PERIS — PE Intelligence",
  description: "Private Equity Research Intelligence System",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-gray-50 text-gray-900 antialiased">
        <Providers>
          <div className="flex h-screen overflow-hidden">
            <Sidebar />
            <main className="flex-1 overflow-y-auto p-8">{children}</main>
          </div>
        </Providers>
      </body>
    </html>
  );
}
