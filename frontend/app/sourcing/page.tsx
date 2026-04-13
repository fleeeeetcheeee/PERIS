"use client";

import { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { getCompanies, getSignals, type Company } from "../lib/api";
import ScoreBadge from "../components/ScoreBadge";
import SignalTypeBadge from "../components/SignalTypeBadge";
import { X, Search } from "lucide-react";

function SlideOver({
  company,
  onClose,
}: {
  company: Company;
  onClose: () => void;
}) {
  const { data: signals } = useQuery({
    queryKey: ["signals", company.id],
    queryFn: () => getSignals(50),
    select: (d) => d.items.filter((s) => s.company_id === company.id),
  });

  return (
    <div className="fixed inset-0 z-40 flex justify-end">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/30" onClick={onClose} />
      {/* Panel */}
      <div className="relative w-full max-w-lg bg-white shadow-2xl overflow-y-auto z-50 p-8">
        <button onClick={onClose} className="absolute top-4 right-4 text-gray-400 hover:text-gray-700">
          <X size={20} />
        </button>

        <h2 className="text-xl font-bold text-gray-900">{company.name}</h2>
        <div className="flex items-center gap-2 mt-2">
          <ScoreBadge score={company.score} />
          {company.sector && (
            <span className="text-xs text-gray-500 bg-gray-100 px-2 py-0.5 rounded">{company.sector}</span>
          )}
          {company.country && (
            <span className="text-xs text-gray-500">{company.country}</span>
          )}
        </div>

        <dl className="mt-6 grid grid-cols-2 gap-4">
          {[
            ["Source", company.source ?? "—"],
            ["Employees", company.employee_count?.toLocaleString() ?? "—"],
            ["Revenue Est.", company.revenue_estimate ? `$${(company.revenue_estimate / 1e6).toFixed(1)}M` : "—"],
            ["Added", company.created_at.slice(0, 10)],
          ].map(([k, v]) => (
            <div key={k}>
              <dt className="text-xs text-gray-500">{k}</dt>
              <dd className="text-sm font-medium text-gray-900 mt-0.5">{v}</dd>
            </div>
          ))}
        </dl>

        {/* Signals */}
        <div className="mt-8">
          <h3 className="text-sm font-semibold text-gray-700 mb-3">Related Signals</h3>
          {!signals || signals.length === 0 ? (
            <p className="text-sm text-gray-400">No signals for this company</p>
          ) : (
            <div className="space-y-3">
              {signals.slice(0, 8).map((s) => (
                <div key={s.id} className="rounded-lg border border-gray-100 p-3">
                  <div className="flex items-center gap-2 mb-1">
                    <SignalTypeBadge type={s.signal_type} />
                    <span className="text-xs text-gray-400">{s.created_at.slice(0, 10)}</span>
                  </div>
                  <p className="text-xs text-gray-700">{s.summary}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default function SourcingPage() {
  const [search, setSearch] = useState("");
  const [sectorFilter, setSectorFilter] = useState("all");
  const [selected, setSelected] = useState<Company | null>(null);

  const { data, isLoading } = useQuery({
    queryKey: ["companies"],
    queryFn: () => getCompanies(500),
  });

  const sectors = useMemo(() => {
    const s = new Set(data?.items.map((c) => c.sector).filter(Boolean) as string[]);
    return ["all", ...Array.from(s).sort()];
  }, [data]);

  const filtered = useMemo(() => {
    return (data?.items ?? []).filter((c) => {
      const matchSearch =
        !search ||
        c.name.toLowerCase().includes(search.toLowerCase()) ||
        (c.sector ?? "").toLowerCase().includes(search.toLowerCase());
      const matchSector = sectorFilter === "all" || c.sector === sectorFilter;
      return matchSearch && matchSector;
    });
  }, [data, search, sectorFilter]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Deal Sourcing</h1>
        <p className="text-sm text-gray-500 mt-1">{data?.count ?? 0} companies tracked</p>
      </div>

      {/* Filters */}
      <div className="flex gap-3">
        <div className="relative flex-1 max-w-sm">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
          <input
            type="text"
            placeholder="Search companies..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full pl-9 pr-4 py-2 text-sm border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        <select
          value={sectorFilter}
          onChange={(e) => setSectorFilter(e.target.value)}
          className="text-sm border border-gray-200 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          {sectors.map((s) => (
            <option key={s} value={s}>
              {s === "all" ? "All Sectors" : s}
            </option>
          ))}
        </select>
      </div>

      {/* Table */}
      <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
        <table className="min-w-full text-sm">
          <thead>
            <tr className="border-b border-gray-100 bg-gray-50">
              {["Name", "Sector", "Country", "Score", "Source", "Added"].map((h) => (
                <th key={h} className="px-4 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wide">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-50">
            {isLoading ? (
              <tr>
                <td colSpan={6} className="px-4 py-8 text-center text-gray-400">Loading...</td>
              </tr>
            ) : filtered.length === 0 ? (
              <tr>
                <td colSpan={6} className="px-4 py-8 text-center text-gray-400">No companies found</td>
              </tr>
            ) : (
              filtered.map((c) => (
                <tr
                  key={c.id}
                  onClick={() => setSelected(c)}
                  className="hover:bg-blue-50 cursor-pointer transition-colors"
                >
                  <td className="px-4 py-3 font-medium text-gray-900">{c.name}</td>
                  <td className="px-4 py-3 text-gray-500">{c.sector ?? "—"}</td>
                  <td className="px-4 py-3 text-gray-500">{c.country ?? "—"}</td>
                  <td className="px-4 py-3">
                    <ScoreBadge score={c.score} />
                  </td>
                  <td className="px-4 py-3 text-gray-500">{c.source ?? "—"}</td>
                  <td className="px-4 py-3 text-gray-400">{c.created_at.slice(0, 10)}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {selected && <SlideOver company={selected} onClose={() => setSelected(null)} />}
    </div>
  );
}
