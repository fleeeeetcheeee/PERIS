"use client";

import { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { getSignals, getCompanies } from "../lib/api";
import SignalTypeBadge from "../components/SignalTypeBadge";

const SIGNAL_TYPES = ["all", "sec_8k", "ma_news", "news", "macro", "sentiment", "monitoring_alert"];

export default function SignalsPage() {
  const [typeFilter, setTypeFilter] = useState("all");

  const { data: signals, isLoading } = useQuery({
    queryKey: ["signals"],
    queryFn: () => getSignals(200),
  });
  const { data: companies } = useQuery({
    queryKey: ["companies"],
    queryFn: () => getCompanies(500),
  });

  const companyMap = Object.fromEntries(
    (companies?.items ?? []).map((c) => [c.id, c.name])
  );

  const filtered = useMemo(() => {
    return (signals?.items ?? []).filter(
      (s) => typeFilter === "all" || s.signal_type === typeFilter
    );
  }, [signals, typeFilter]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Signals</h1>
        <p className="text-sm text-gray-500 mt-1">
          {signals?.count ?? 0} signals — live feed, refreshes every 30s
        </p>
      </div>

      {/* Filter tabs */}
      <div className="flex flex-wrap gap-2">
        {SIGNAL_TYPES.map((t) => (
          <button
            key={t}
            onClick={() => setTypeFilter(t)}
            className={`text-xs px-3 py-1.5 rounded-full border font-medium transition-colors ${
              typeFilter === t
                ? "bg-blue-600 text-white border-blue-600"
                : "bg-white text-gray-600 border-gray-200 hover:border-blue-300"
            }`}
          >
            {t === "all" ? "All Types" : t.replace(/_/g, " ").toUpperCase()}
          </button>
        ))}
      </div>

      {/* Feed */}
      <div className="bg-white rounded-xl border border-gray-200 divide-y divide-gray-100">
        {isLoading ? (
          <p className="px-6 py-8 text-center text-gray-400">Loading signals...</p>
        ) : filtered.length === 0 ? (
          <p className="px-6 py-8 text-center text-gray-400">No signals match this filter</p>
        ) : (
          filtered.map((s) => (
            <div key={s.id} className="px-6 py-4 flex items-start gap-4">
              <div className="shrink-0 pt-0.5">
                <SignalTypeBadge type={s.signal_type} />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-0.5">
                  <span className="text-xs font-medium text-gray-700">
                    {companyMap[s.company_id] ?? `Company #${s.company_id}`}
                  </span>
                  <span className="text-xs text-gray-400">
                    {s.created_at.slice(0, 16).replace("T", " ")}
                  </span>
                </div>
                <p className="text-sm text-gray-700">{s.summary}</p>
              </div>
              {s.confidence !== null && (
                <div className="shrink-0 text-right">
                  <span className="text-xs font-semibold text-gray-500">
                    {(s.confidence * 100).toFixed(0)}%
                  </span>
                  <p className="text-xs text-gray-400">conf.</p>
                </div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
