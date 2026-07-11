/**
 * Typed fetch functions for Screen 1 (Live Feed) — GET/POST /api/live-feed/*.
 * Backed by src/api/routers/live_feed.py (L1 only — see that file's module
 * docstring for the architectural boundary this screen proves).
 */

import type {
  LiveFeedNewsResponse,
  LiveFeedWeatherResponse,
  IngestStatusResponse,
  LiveFeedLogsResponse,
  LiveFeedGanttResponse,
  RefreshTriggeredResponse,
} from "../types/liveFeed";
import { API_BASE_URL } from "./config";

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`);
  if (!res.ok) throw new Error(`${path} failed: ${res.status}`);
  return res.json();
}

export const fetchLiveFeedNews = () =>
  getJSON<LiveFeedNewsResponse>("/api/live-feed/news");

export const fetchLiveFeedWeather = () =>
  getJSON<LiveFeedWeatherResponse>("/api/live-feed/weather");

export const fetchIngestStatus = () =>
  getJSON<IngestStatusResponse>("/api/live-feed/ingest-status");

export const fetchLiveFeedLogs = (runId?: string) =>
  getJSON<LiveFeedLogsResponse>(
    `/api/live-feed/logs${runId ? `?run_id=${runId}` : ""}`,
  );

export const fetchLiveFeedGantt = (runId?: string) =>
  getJSON<LiveFeedGanttResponse>(
    `/api/live-feed/gantt${runId ? `?run_id=${runId}` : ""}`,
  );

export async function triggerLiveFeedRefresh(): Promise<RefreshTriggeredResponse> {
  const res = await fetch(`${API_BASE_URL}/api/live-feed/refresh`, { method: "POST" });
  if (!res.ok) throw new Error(`refresh failed: ${res.status}`);
  return res.json();
}
