/**
 * Screen 1 (Live Feed) — Weather column, 6 fixed fab-hub cities.
 * Real data source: GET /api/live-feed/weather → live_weather_ingest,
 * populated by DataIngestionAgent's Open-Meteo connector (L1 only).
 * Always renders all 6 configured hubs (Hsinchu, Osaka, Austin, Shanghai,
 * Singapore, Rotterdam) even when a hub has no ingested row yet — see the
 * backend endpoint's docstring for why. `is_trigger_hub` is server-computed;
 * never recompute the trigger threshold here.
 */
import { useLiveFeedWeather } from "../../hooks/useLiveFeed";
import type { WeatherHub } from "../../types/liveFeed";

// Cosmetic-only lookup (flag emoji, weather icon) — not part of the API
// contract; safe to extend without touching the backend.
const HUB_FLAGS: Record<string, string> = {
  Hsinchu: "🇹🇼",
  Osaka: "🇯🇵",
  Austin: "🇺🇸",
  Shanghai: "🇨🇳",
  Singapore: "🇸🇬",
  Rotterdam: "🇳🇱",
};

function timeAgo(timestamp: string | null): string {
  if (!timestamp) return "—";
  const then = new Date(timestamp).getTime();
  if (Number.isNaN(then)) return "—";
  const minutes = Math.max(0, Math.round((Date.now() - then) / 60000));
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ${minutes % 60}m ago`;
}

function weatherIcon(code: number | null): string {
  if (code == null) return "—";
  if (code >= 95) return "⛈️";
  if (code >= 80) return "🌧️";
  if (code >= 61) return "🌦️";
  if (code >= 2) return "☁️";
  return "⛅";
}

function severityTextClass(severity: number): string {
  if (severity > 7) return "text-risk-critical";
  if (severity > 4) return "text-risk-medium";
  return "text-risk-low";
}

function severityBgClass(severity: number): string {
  if (severity > 7) return "bg-risk-critical";
  if (severity > 4) return "bg-risk-medium";
  return "bg-risk-low";
}

function WeatherCard({ hub }: { hub: WeatherHub }) {
  const hasData = hub.raw_severity_score != null;

  return (
    <div
      className={`p-2.5 rounded-lg bg-background ${
        hub.is_trigger_hub
          ? "border-[1.5px] border-risk-critical/30 shadow-[0_0_14px_rgba(239,68,68,0.09)]"
          : "border border-border"
      }`}
    >
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-xs font-semibold text-foreground">
          {HUB_FLAGS[hub.hub_city] ?? ""} {hub.hub_city}
        </span>
        {hub.is_trigger_hub && (
          <span className="text-[9px] font-mono text-risk-critical border border-risk-critical/30 px-1 rounded">
            TRIGGER
          </span>
        )}
      </div>

      {!hasData ? (
        <div className="text-[10px] text-muted-foreground py-2">No data yet</div>
      ) : (
        <>
          <div className="text-xl mb-1.5">{weatherIcon(hub.weather_code)}</div>
          <div className="space-y-0.5 text-[10px]">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Wind</span>
              <span className="font-mono text-muted-strong">
                {hub.wind_speed_kmh ?? "—"} km/h
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Precip</span>
              <span className="font-mono text-muted-strong">
                {hub.precipitation_mm ?? "—"} mm
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Temp</span>
              <span className="font-mono text-muted-strong">
                {hub.temperature_c ?? "—"}°C
              </span>
            </div>
          </div>
          <div className="mt-2">
            <div className="flex justify-between text-[9px] mb-1">
              <span className="text-muted-foreground">raw_severity</span>
              <span
                className={`font-mono font-semibold ${severityTextClass(hub.raw_severity_score!)}`}
              >
                {hub.raw_severity_score}/10
              </span>
            </div>
            <div className="h-1 rounded-full overflow-hidden bg-border">
              <div
                className={`h-full rounded-full ${severityBgClass(hub.raw_severity_score!)}`}
                style={{ width: `${hub.raw_severity_score! * 10}%` }}
              />
            </div>
          </div>
        </>
      )}
    </div>
  );
}

export function WeatherPanel() {
  const { data, isLoading, isError } = useLiveFeedWeather();

  return (
    <div className="flex flex-col overflow-hidden rounded-lg bg-card border border-border">
      <div className="px-3 py-2.5 flex items-center justify-between shrink-0 border-b border-border">
        <span className="text-xs font-semibold text-foreground">
          Open-Meteo — 6 Fab-Hub Cities
        </span>
        {data && (
          <span className="text-[10px] font-mono text-muted-foreground">
            updated {timeAgo(data.fetched_at)}
          </span>
        )}
      </div>
      <div className="flex-1 overflow-y-auto p-2 grid grid-cols-2 gap-2 content-start">
        {isLoading && (
          <div className="col-span-2 text-xs text-muted-foreground px-1">
            Loading weather…
          </div>
        )}
        {isError && (
          <div className="col-span-2 text-xs text-risk-critical px-1">
            Could not load weather feed. Check backend logs.
          </div>
        )}
        {data?.hubs.map((hub) => (
          <WeatherCard key={hub.hub_city} hub={hub} />
        ))}
      </div>
    </div>
  );
}
