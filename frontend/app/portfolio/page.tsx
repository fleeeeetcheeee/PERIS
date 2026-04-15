"use client";

import { useQuery } from "@tanstack/react-query";
import { getCompanies, getPortfolio } from "../lib/api";
import { LineChart, Line, ResponsiveContainer, Tooltip } from "recharts";

function StatusBadge({ status }: { status: "On Track" | "Watch" | "Alert" }) {
  const cls =
    status === "On Track"
      ? "bg-green-100 text-green-800"
      : status === "Watch"
      ? "bg-yellow-100 text-yellow-800"
      : "bg-red-100 text-red-800";
  return (
    <span className={`text-xs font-semibold px-2 py-0.5 rounded ${cls}`}>{status}</span>
  );
}

function deriveStatus(score: number | null): "On Track" | "Watch" | "Alert" {
  if (score === null) return "Watch";
  if (score >= 70) return "On Track";
  if (score >= 45) return "Watch";
  return "Alert";
}

export default function PortfolioPage() {
  const { data: companies } = useQuery({
    queryKey: ["companies"],
    queryFn: () => getCompanies(500),
  });
  const { data: kpis } = useQuery({
    queryKey: ["portfolio"],
    queryFn: () => getPortfolio(),
  });

  // Portfolio companies = scored >= 60
  const portfolioCompanies = (companies?.items ?? []).filter(
    (c) => c.score !== null && c.score >= 60 && c.name !== "_MACRO_DATA_"
  );

  // Group KPIs by company_id
  const kpiMap: Record<number, { value: number; metric_name: string }[]> = {};
  kpis?.items.forEach((k) => {
    if (!kpiMap[k.company_id]) kpiMap[k.company_id] = [];
    kpiMap[k.company_id].push({ value: k.value, metric_name: k.metric_name });
  });

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Portfolio</h1>
        <p className="text-sm text-gray-500 mt-1">
          {portfolioCompanies.length} companies with score ≥ 60
        </p>
      </div>

      {portfolioCompanies.length === 0 ? (
        <div className="bg-white rounded-xl border border-gray-200 p-12 text-center">
          <p className="text-gray-400">No portfolio companies yet. Score companies from the Sourcing page.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {portfolioCompanies.map((c) => {
            const companyKpis = kpiMap[c.id] ?? [];
            const sparkData = companyKpis.slice(-8).map((k, i) => ({ i, value: k.value }));
            const status = deriveStatus(c.score);

            return (
              <div key={c.id} className="bg-white rounded-xl border border-gray-200 p-5">
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0 flex-1 overflow-hidden">
                    <p className="font-semibold text-gray-900 truncate">{c.name}</p>
                    <p className="text-xs text-gray-500 mt-0.5 truncate">{c.sector ?? "Unknown sector"}</p>
                  </div>
                  <div className="shrink-0">
                    <StatusBadge status={status} />
                  </div>
                </div>

                {/* Score */}
                <div className="mt-3 flex items-center gap-2">
                  <div className="flex-1 bg-gray-100 rounded-full h-1.5">
                    <div
                      className={`h-1.5 rounded-full ${
                        status === "On Track"
                          ? "bg-green-500"
                          : status === "Watch"
                          ? "bg-yellow-400"
                          : "bg-red-500"
                      }`}
                      style={{ width: `${c.score ?? 0}%` }}
                    />
                  </div>
                  <span className="text-xs font-semibold text-gray-700">
                    {c.score?.toFixed(0) ?? "—"}
                  </span>
                </div>

                {/* Sparkline */}
                {sparkData.length > 1 ? (
                  <div className="mt-4 h-14">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={sparkData}>
                        <Line
                          type="monotone"
                          dataKey="value"
                          stroke="#2563eb"
                          strokeWidth={1.5}
                          dot={false}
                        />
                        <Tooltip
                          contentStyle={{ fontSize: 11 }}
                          formatter={(v) => [typeof v === "number" ? v.toFixed(2) : v, "value"]}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <p className="mt-4 text-xs text-gray-400">No KPI data</p>
                )}

                {/* KPI pills */}
                {companyKpis.length > 0 && (
                  <div className="mt-3 flex flex-wrap gap-1">
                    {companyKpis.slice(0, 3).map((k, i) => (
                      <span key={i} className="text-xs bg-gray-100 text-gray-600 px-2 py-0.5 rounded">
                        {k.metric_name}: {k.value.toFixed(2)}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
