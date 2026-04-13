"use client";

import { useQuery } from "@tanstack/react-query";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import Link from "next/link";
import { getCompanies, getSignals, getPipeline, getPortfolio } from "../lib/api";
import SignalTypeBadge from "../components/SignalTypeBadge";

function MetricCard({
  label,
  value,
  sub,
  href,
}: {
  label: string;
  value: number | string;
  sub?: string;
  href: string;
}) {
  return (
    <Link
      href={href}
      className="bg-white rounded-xl border border-gray-200 p-6 hover:border-blue-300 hover:shadow-sm transition-all cursor-pointer block"
    >
      <p className="text-sm text-gray-500 font-medium">{label}</p>
      <p className="text-3xl font-bold text-gray-900 mt-1">{value}</p>
      {sub && <p className="text-xs text-gray-400 mt-1">{sub}</p>}
    </Link>
  );
}

export default function DashboardPage() {
  const { data: companies } = useQuery({
    queryKey: ["companies"],
    queryFn: () => getCompanies(200),
  });
  const { data: signals } = useQuery({
    queryKey: ["signals"],
    queryFn: () => getSignals(100),
  });
  const { data: pipeline } = useQuery({
    queryKey: ["pipeline"],
    queryFn: () => getPipeline(),
  });
  const { data: portfolio } = useQuery({
    queryKey: ["portfolio"],
    queryFn: () => getPortfolio(),
  });

  const totalCompanies = companies?.count ?? 0;
  const pipelineDeals = pipeline?.count ?? 0;
  const portfolioKpis = portfolio?.count ?? 0;

  const today = new Date().toISOString().slice(0, 10);
  const signalsToday =
    signals?.items.filter((s) => s.created_at.slice(0, 10) === today).length ?? 0;

  // Build chart data: companies scored per day
  const scoredByDay: Record<string, number> = {};
  companies?.items
    .filter((c) => c.score !== null)
    .forEach((c) => {
      const day = c.created_at.slice(0, 10);
      scoredByDay[day] = (scoredByDay[day] ?? 0) + 1;
    });
  const chartData = Object.entries(scoredByDay)
    .sort(([a], [b]) => a.localeCompare(b))
    .slice(-14)
    .map(([date, count]) => ({ date: date.slice(5), count }));

  const recentSignals = signals?.items.slice(0, 10) ?? [];

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
        <p className="text-sm text-gray-500 mt-1">Live overview — refreshes every 30s</p>
      </div>

      {/* Metric cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard label="Companies Tracked" value={totalCompanies} sub="all sources" href="/sourcing" />
        <MetricCard label="Pipeline Deals" value={pipelineDeals} sub="active stages" href="/pipeline" />
        <MetricCard label="KPI Records" value={portfolioKpis} sub="portfolio metrics" href="/portfolio" />
        <MetricCard label="Signals Today" value={signalsToday} sub={today} href="/signals" />
      </div>

      {/* Chart */}
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <h2 className="text-base font-semibold text-gray-800 mb-4">Companies Scored (Last 14 Days)</h2>
        {chartData.length > 0 ? (
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="date" tick={{ fontSize: 12 }} />
              <YAxis tick={{ fontSize: 12 }} allowDecimals={false} />
              <Tooltip />
              <Line
                type="monotone"
                dataKey="count"
                stroke="#2563eb"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <p className="text-sm text-gray-400 py-10 text-center">No scored companies yet</p>
        )}
      </div>

      {/* Recent signals */}
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <h2 className="text-base font-semibold text-gray-800 mb-4">Recent Signals</h2>
        {recentSignals.length === 0 ? (
          <p className="text-sm text-gray-400">No signals yet</p>
        ) : (
          <div className="divide-y divide-gray-100">
            {recentSignals.map((s) => (
              <div key={s.id} className="py-3 flex items-start gap-3">
                <SignalTypeBadge type={s.signal_type} />
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-gray-800 truncate">{s.summary}</p>
                  <p className="text-xs text-gray-400 mt-0.5">{s.created_at.slice(0, 16).replace("T", " ")}</p>
                </div>
                {s.confidence !== null && (
                  <span className="text-xs text-gray-400 shrink-0">
                    {(s.confidence * 100).toFixed(0)}%
                  </span>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
