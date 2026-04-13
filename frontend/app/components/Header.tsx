"use client";

import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { runIngest, getIngestStatus } from "../lib/api";
import { RefreshCw } from "lucide-react";

export default function Header() {
  const [done, setDone] = useState(false);

  const { data: status } = useQuery({
    queryKey: ["ingest-status"],
    queryFn: getIngestStatus,
    refetchInterval: 3000,
  });

  const mutation = useMutation({
    mutationFn: runIngest,
    onSuccess: () => setDone(true),
  });

  const isRunning = status?.running || mutation.isPending;

  return (
    <header className="h-14 flex-shrink-0 bg-white border-b border-gray-200 flex items-center justify-end px-8">
      <button
        onClick={() => { setDone(false); mutation.mutate(); }}
        disabled={isRunning}
        className="flex items-center gap-2 px-3 py-1.5 text-sm font-medium rounded-lg border border-gray-200 text-gray-700 hover:bg-gray-50 disabled:opacity-50 transition-colors"
      >
        <RefreshCw size={13} className={isRunning ? "animate-spin" : ""} />
        {isRunning ? "Ingesting…" : done ? "Ingest Complete" : "Run Ingest"}
      </button>
    </header>
  );
}
