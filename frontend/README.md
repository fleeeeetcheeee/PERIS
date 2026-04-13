# PERIS Frontend

Next.js 14 dashboard for the PERIS PE Intelligence System.

## Prerequisites

- Node.js 18+
- PERIS FastAPI backend running on `http://localhost:8000`

## Run

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Pages

| Route | Description |
|-------|-------------|
| `/dashboard` | Metric cards, scored-companies chart, recent signals |
| `/sourcing` | Searchable company table with slide-over detail panel |
| `/pipeline` | Kanban board with drag-and-drop stage management |
| `/portfolio` | Portfolio company cards with KPI sparklines |
| `/signals` | Live signal feed filterable by type |
| `/reports` | PDF report list with download links and on-demand generation |

## Environment

`.env.local` (already included):
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```
