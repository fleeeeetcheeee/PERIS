import axios from "axios";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export const api = axios.create({ baseURL: BASE });

// ── Types ────────────────────────────────────────────────────────────────────

export interface Company {
  id: number;
  name: string;
  sector: string | null;
  country: string | null;
  employee_count: number | null;
  revenue_estimate: number | null;
  source: string | null;
  score: number | null;
  created_at: string;
}

export interface PipelineStage {
  id: number;
  company_id: number;
  stage: string;
  owner: string | null;
  notes: string | null;
  updated_at: string;
}

export interface PortfolioKPI {
  id: number;
  company_id: number;
  metric_name: string;
  value: number;
  period: string | null;
  recorded_at: string;
}

export interface Signal {
  id: number;
  company_id: number;
  signal_type: string;
  summary: string;
  raw_data: Record<string, unknown> | null;
  confidence: number | null;
  created_at: string;
}

export interface Report {
  filename: string;
  size_bytes: number;
  path: string;
}

export interface ListResponse<T> {
  items: T[];
  count: number;
}

// ── Fetch functions ───────────────────────────────────────────────────────────

export async function getTopCompanies(limit = 10, minScore = 60): Promise<ListResponse<Company>> {
  const { data } = await api.get<ListResponse<Company>>("/companies/top", {
    params: { limit, min_score: minScore },
  });
  return data;
}

export async function getCompanies(limit = 200): Promise<ListResponse<Company>> {
  const { data } = await api.get<ListResponse<Company>>("/companies/", { params: { limit } });
  return data;
}

export async function getCompany(id: number): Promise<Company> {
  const { data } = await api.get<Company>(`/companies/${id}`);
  return data;
}

export async function getPipeline(limit = 500): Promise<ListResponse<PipelineStage>> {
  const { data } = await api.get<ListResponse<PipelineStage>>("/pipeline/", { params: { limit } });
  return data;
}

export async function getPortfolio(limit = 500): Promise<ListResponse<PortfolioKPI>> {
  const { data } = await api.get<ListResponse<PortfolioKPI>>("/portfolio/", { params: { limit } });
  return data;
}

export async function getSignals(limit = 100): Promise<ListResponse<Signal>> {
  const { data } = await api.get<ListResponse<Signal>>("/signals/", { params: { limit } });
  return data;
}

export async function getReports(): Promise<ListResponse<Report>> {
  const { data } = await api.get<ListResponse<Report>>("/reports/");
  return data;
}

export async function generateReport(): Promise<{ status: string; pdf_path: string }> {
  const { data } = await api.post("/reports/generate");
  return data;
}

export async function getThesis(): Promise<string> {
  const { data } = await api.get<{ thesis: string }>("/thesis/");
  return data.thesis;
}

export async function saveThesis(thesis: string): Promise<void> {
  await api.post("/thesis/", { thesis });
}

export async function runIngest(): Promise<{ status: string }> {
  const { data } = await api.post<{ status: string }>("/ingest/run");
  return data;
}

export async function getIngestStatus(): Promise<{ running: boolean }> {
  const { data } = await api.get<{ running: boolean }>("/ingest/status");
  return data;
}

export async function updatePipelineStage(
  id: number,
  payload: Partial<Pick<PipelineStage, "stage" | "owner" | "notes">>
): Promise<PipelineStage> {
  const { data } = await api.patch<PipelineStage>(`/pipeline/${id}`, payload);
  return data;
}
