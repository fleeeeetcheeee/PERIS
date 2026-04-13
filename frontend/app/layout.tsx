import type { Metadata } from "next";
import "./globals.css";
import Sidebar from "./components/Sidebar";
import Header from "./components/Header";
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
            <div className="flex-1 flex flex-col overflow-hidden">
              <Header />
              <main className="flex-1 overflow-y-auto p-8">{children}</main>
            </div>
          </div>
        </Providers>
      </body>
    </html>
  );
}
