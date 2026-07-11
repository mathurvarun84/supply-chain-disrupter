/**
 * TypeScript types for Screen 1 (Live Feed) — mirrors src/api/live_feed_schemas.py
 * 1:1. Keep both files in sync when adding/changing fields.
 */

export interface NewsHeadline {
  hub_city: string | null;
  hub_country: string | null;
  supplier_country: string | null;
  headline: string;
  published_at: string | null;
  relevance_score: number | null;
  query_term: string | null;
  source_feed: string | null;
  score_tier: "low" | "medium" | "high";
}

export interface LiveFeedNewsResponse {
  run_id: string | null;
  count: number;
  fetched_at: string | null;
  items: NewsHeadline[];
}

export interface WeatherHub {
  hub_city: string;
  wind_speed_kmh: number | null;
  precipitation_mm: number | null;
  weather_code: number | null;
  temperature_c: number | null;
  raw_severity_score: number | null;
  is_trigger_hub: boolean;
  fetched_at_utc: string | null;
}

export interface LiveFeedWeatherResponse {
  run_id: string | null;
  fetched_at: string | null;
  hubs: WeatherHub[];
}

export interface IngestRunLogRow {
  run_ts_utc: string;
  source: string;
  status: string;
  rows_inserted: number;
  rows_skipped: number;
  duration_ms: number | null;
  error_detail: string | null;
}

export interface IngestStatusResponse {
  is_running: boolean;
  last_run: IngestRunLogRow | null;
  recent_runs: IngestRunLogRow[];
}

export interface RefreshTriggeredResponse {
  status: "started" | "skipped_already_running";
  triggered_at: string;
}

export interface LogLine {
  level: string;
  text: string;
  tab: number;
  source: "real" | "stub";
}

export interface LiveFeedLogsResponse {
  lines: LogLine[];
}

export interface GanttBar {
  id: string;
  start: number;
  dur: number;
  color: string;
  source: "real" | "stub";
}

export interface LiveFeedGanttResponse {
  bars: GanttBar[];
}
