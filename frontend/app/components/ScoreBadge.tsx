export default function ScoreBadge({ score }: { score: number | null }) {
  if (score === null) return <span className="text-gray-400 text-xs">—</span>;

  const color =
    score >= 75
      ? "bg-green-100 text-green-800"
      : score >= 50
      ? "bg-yellow-100 text-yellow-800"
      : "bg-red-100 text-red-800";

  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold ${color}`}>
      {score.toFixed(0)}
    </span>
  );
}
