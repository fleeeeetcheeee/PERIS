const COLORS: Record<string, string> = {
  sec_8k: "bg-purple-100 text-purple-800",
  ma_news: "bg-blue-100 text-blue-800",
  news: "bg-gray-100 text-gray-700",
  macro: "bg-orange-100 text-orange-800",
  sentiment: "bg-pink-100 text-pink-800",
  monitoring_alert: "bg-red-100 text-red-800",
};

export default function SignalTypeBadge({ type }: { type: string }) {
  const cls = COLORS[type] ?? "bg-gray-100 text-gray-600";
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${cls}`}>
      {type.replace(/_/g, " ").toUpperCase()}
    </span>
  );
}
