"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { getPipeline, getCompanies, updatePipelineStage, type PipelineStage } from "../lib/api";
import ScoreBadge from "../components/ScoreBadge";

const STAGES = ["Target", "Contacted", "LOI", "Diligence", "Closed"];

export default function PipelinePage() {
  const qc = useQueryClient();
  const [dragId, setDragId] = useState<number | null>(null);

  const { data: pipeline } = useQuery({
    queryKey: ["pipeline"],
    queryFn: () => getPipeline(),
  });
  const { data: companies } = useQuery({
    queryKey: ["companies"],
    queryFn: () => getCompanies(500),
  });

  const mutation = useMutation({
    mutationFn: ({ id, stage }: { id: number; stage: string }) =>
      updatePipelineStage(id, { stage }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["pipeline"] }),
  });

  const companyMap = Object.fromEntries(
    (companies?.items ?? []).map((c) => [c.id, c])
  );

  const stageMap: Record<string, PipelineStage[]> = Object.fromEntries(
    STAGES.map((s) => [s, []])
  );
  pipeline?.items.forEach((ps) => {
    const key = STAGES.find((s) => s.toLowerCase() === ps.stage.toLowerCase()) ?? ps.stage;
    if (!stageMap[key]) stageMap[key] = [];
    stageMap[key].push(ps);
  });

  const handleDrop = (stage: string) => {
    if (dragId === null) return;
    mutation.mutate({ id: dragId, stage: stage.toLowerCase() });
    setDragId(null);
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Pipeline</h1>
        <p className="text-sm text-gray-500 mt-1">Drag cards to move deals between stages</p>
      </div>

      <div className="flex gap-4 overflow-x-auto pb-4">
        {STAGES.map((stage) => {
          const cards = stageMap[stage] ?? [];
          return (
            <div
              key={stage}
              className="flex-shrink-0 w-60"
              onDragOver={(e) => e.preventDefault()}
              onDrop={() => handleDrop(stage)}
            >
              {/* Column header */}
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-semibold text-gray-700">{stage}</h3>
                <span className="text-xs bg-gray-200 text-gray-600 px-1.5 py-0.5 rounded-full">
                  {cards.length}
                </span>
              </div>

              {/* Drop zone */}
              <div className="space-y-2 min-h-[120px] rounded-xl bg-gray-100 p-2">
                {cards.map((ps) => {
                  const co = companyMap[ps.company_id];
                  return (
                    <div
                      key={ps.id}
                      draggable
                      onDragStart={() => setDragId(ps.id)}
                      className="bg-white rounded-lg border border-gray-200 p-3 cursor-grab active:cursor-grabbing shadow-sm hover:shadow-md transition-shadow"
                    >
                      <p className="text-sm font-medium text-gray-900 truncate">
                        {co?.name ?? `Company #${ps.company_id}`}
                      </p>
                      <div className="flex items-center justify-between mt-2">
                        <ScoreBadge score={co?.score ?? null} />
                        <span className="text-xs text-gray-400">
                          {ps.updated_at.slice(0, 10)}
                        </span>
                      </div>
                      {ps.owner && (
                        <p className="text-xs text-gray-400 mt-1">Owner: {ps.owner}</p>
                      )}
                    </div>
                  );
                })}
                {cards.length === 0 && (
                  <p className="text-xs text-gray-400 text-center py-4">Drop here</p>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
