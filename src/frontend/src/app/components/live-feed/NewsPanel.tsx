/**
 * Screen 1 (Live Feed) — News column.
 * Real data source: GET /api/live-feed/news → live_news_ingest, populated by
 * DataIngestionAgent's NewsRSSConnector (L1 only — see ARCHITECTURE.md).
 * Despite the panel title's original "Google News RSS" branding, the
 * connector already blends multiple RSS sources (Google News RSS primary,
 * Reuters technology RSS as a per-query fallback — see
 * src/agents/data_ingestion_agent.py / ingestion_connectors.py), so the
 * panel is labelled "News RSS" and each headline's `source_feed` reflects
 * its true origin instead of a single hardcoded brand name.
 * Polls every 15s; "Refresh Live Data" (RefreshControl.tsx) triggers a fresh
 * ingestion batch. Layout/classNames copied from the original mockup body in
 * _reference/App.mockup.tsx — do not restyle, only the data source changed.
 */
import { useLiveFeedNews } from "../../hooks/useLiveFeed";
import type { NewsHeadline } from "../../types/liveFeed";

// Mockup grouped news by query type (hub city / hub country / supplier
// country); the real rows carry that same distinction via which of the
// three columns is non-null, so we regroup client-side to match.
function groupByQueryType(items: NewsHeadline[]) {
  const groups: { group: string; items: NewsHeadline[] }[] = [
    { group: "Hub City Queries", items: [] },
    { group: "Hub Country Queries", items: [] },
    { group: "Supplier Country Queries", items: [] },
  ];
  for (const item of items) {
    if (item.hub_city) groups[0].items.push(item);
    else if (item.hub_country) groups[1].items.push(item);
    else if (item.supplier_country) groups[2].items.push(item);
  }
  return groups.filter((g) => g.items.length > 0);
}

function tagFor(item: NewsHeadline): string {
  return item.hub_city ?? item.hub_country ?? item.supplier_country ?? "—";
}

// score_tier is server-computed (src/api/routers/live_feed.py::_score_tier);
// this map only assigns the chip color, it never recomputes the threshold.
const TIER_CLASS: Record<NewsHeadline["score_tier"], string> = {
  high: "bg-risk-critical/20 text-risk-critical",
  medium: "bg-risk-high/20 text-risk-high",
  low: "bg-risk-medium/20 text-risk-medium",
};

function timeAgo(publishedAt: string | null): string {
  if (!publishedAt) return "—";
  const then = new Date(publishedAt).getTime();
  if (Number.isNaN(then)) return "—";
  const minutes = Math.max(0, Math.round((Date.now() - then) / 60000));
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ${minutes % 60}m ago`;
}

export function NewsPanel() {
  const { data, isLoading, isError } = useLiveFeedNews();
  const groups = data ? groupByQueryType(data.items) : [];

  return (
    <div className="flex flex-col overflow-hidden rounded-lg bg-card border border-border">
      <div className="px-3 py-2.5 flex items-center justify-between shrink-0 border-b border-border">
        <span className="text-xs font-semibold text-foreground">
          News RSS — 14 parallel queries
        </span>
        {data && (
          <span className="text-[10px] font-mono text-muted-foreground">
            {data.count} new · updated {timeAgo(data.fetched_at)} · run_id{" "}
            {data.run_id?.slice(0, 8) ?? "—"}
          </span>
        )}
      </div>

      <div className="flex-1 overflow-y-auto p-2 space-y-3">
        {isLoading && (
          <div className="text-xs text-muted-foreground px-1">Loading news…</div>
        )}
        {isError && (
          <div className="text-xs text-risk-critical px-1">
            Could not load news feed. Check backend logs.
          </div>
        )}
        {data && data.items.length === 0 && (
          <div className="text-xs text-muted-foreground px-1">
            No news ingested yet — click "Refresh Live Data" above.
          </div>
        )}

        {groups.map((g) => (
          <div key={g.group}>
            <div className="text-[9px] font-semibold text-muted-foreground uppercase tracking-wider mb-1.5 px-1">
              {g.group}
            </div>
            {g.items.map((item, i) => (
              <div
                key={i}
                className="mb-1.5 p-2 rounded bg-background border border-border"
              >
                <div className="text-[11px] text-foreground leading-snug mb-1.5">
                  {item.headline}
                </div>
                <div className="flex items-center gap-1.5 flex-wrap">
                  <span className="text-[9px] text-muted-foreground">
                    {item.source_feed ?? "—"}
                  </span>
                  <span className="text-[9px] text-border2">·</span>
                  <span className="text-[9px] font-mono text-primary">{tagFor(item)}</span>
                  <span className="text-[9px] text-border2">·</span>
                  <span className="text-[9px] text-muted-foreground">
                    {timeAgo(item.published_at)}
                  </span>
                  <span
                    className={`ml-auto text-[9px] font-mono px-1.5 py-0.5 rounded ${TIER_CLASS[item.score_tier]}`}
                  >
                    {item.relevance_score?.toFixed(2) ?? "—"}
                  </span>
                </div>
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}
