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
 * The headline list also auto-scrolls on a slow continuous loop (like a
 * news ticker) so the panel visibly "runs" on its own; pauses on
 * hover/touch and is skipped entirely under prefers-reduced-motion.
 */
import { useEffect, useRef, useState, type RefObject } from "react";
import { useLiveFeedNews } from "../../hooks/useLiveFeed";
import type { NewsHeadline } from "../../types/liveFeed";
import { prefersReducedMotion } from "../../utils/animation";

// Ticker-like auto-scroll pace so the list reads as "always running" (per
// user request — like a live news ticker). Bumped up from an initial 16 —
// that was imperceptible against typical panel content height; this is
// still slow enough to read headlines mid-scroll, but visibly moving.
const AUTO_SCROLL_PX_PER_SEC = 35;
// How long a hover/touch pause holds before the ticker resumes.
const RESUME_DELAY_MS = 2500;

// Continuously scrolls `el` downward, looping back to the top at the end.
// Pauses while `pausedRef.current` is true (hover/touch) and is a no-op
// under reduced motion. Re-measures scrollHeight every frame rather than
// once, so it keeps working across React Query's 15s data refreshes
// without needing to be restarted.
function useAutoScrollTicker(pausedRef: RefObject<boolean>) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (prefersReducedMotion()) return;
    const el = containerRef.current;
    if (!el) return;

    let raf = 0;
    let last = performance.now();

    const tick = (now: number) => {
      const dt = (now - last) / 1000;
      last = now;
      if (!pausedRef.current) {
        const maxScroll = el.scrollHeight - el.clientHeight;
        if (maxScroll > 0) {
          const next = el.scrollTop + AUTO_SCROLL_PX_PER_SEC * dt;
          el.scrollTop = next >= maxScroll ? 0 : next;
        }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return containerRef;
}

// News RSS refetches every 15s (see useLiveFeedNews in hooks/useLiveFeed.ts,
// which this presentation-only pass must not touch) — kept in lockstep with
// that literal by hand; update both if the poll interval ever changes.
const NEWS_POLL_MS = 15_000;

// NewsHeadline carries no stable id from the API, so "genuinely new
// headline" is approximated with a composite of fields unlikely to repeat
// across a poll window.
function newsItemKey(item: NewsHeadline): string {
  return `${item.headline}|${item.published_at ?? ""}|${item.source_feed ?? ""}`;
}

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

// Thin countdown bar in the panel header — purely visual, driven by a local
// tick against dataUpdatedAt, never causes an early/duplicate fetch.
function RefreshCountdown({ dataUpdatedAt }: { dataUpdatedAt: number }) {
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 250);
    return () => clearInterval(id);
  }, []);

  const elapsed = dataUpdatedAt ? now - dataUpdatedAt : 0;
  const remainingFrac = Math.max(0, Math.min(1, 1 - elapsed / NEWS_POLL_MS));

  return (
    <div
      className="w-10 h-1 rounded-full overflow-hidden bg-border shrink-0"
      title="Next refresh countdown"
    >
      <div
        className="h-full rounded-full bg-primary transition-[width] duration-300 ease-linear motion-reduce:transition-none"
        style={{ width: `${remainingFrac * 100}%` }}
      />
    </div>
  );
}

export function NewsPanel() {
  const { data, isLoading, isError, dataUpdatedAt } = useLiveFeedNews();
  const groups = data ? groupByQueryType(data.items) : [];

  const seenIds = useRef<Set<string>>(new Set());
  useEffect(() => {
    if (!data) return;
    for (const item of data.items) seenIds.current.add(newsItemKey(item));
  }, [data]);

  // Ticker pauses on hover (desktop) or touch (mobile) so a reader isn't
  // fighting the scroll to read a headline, then resumes on its own.
  const pausedRef = useRef(false);
  const resumeTimer = useRef<number | undefined>(undefined);
  const pause = () => {
    window.clearTimeout(resumeTimer.current);
    pausedRef.current = true;
  };
  const resumeNow = () => {
    window.clearTimeout(resumeTimer.current);
    pausedRef.current = false;
  };
  const resumeAfterDelay = () => {
    window.clearTimeout(resumeTimer.current);
    resumeTimer.current = window.setTimeout(() => {
      pausedRef.current = false;
    }, RESUME_DELAY_MS);
  };
  const scrollRef = useAutoScrollTicker(pausedRef);

  return (
    <div className="flex flex-col overflow-hidden rounded-lg bg-card border border-border">
      <div className="px-3 py-2.5 flex items-center justify-between shrink-0 border-b border-border">
        <span className="text-xs font-semibold text-foreground">
          News RSS — 14 parallel queries
        </span>
        <div className="flex items-center gap-2">
          {data && <RefreshCountdown dataUpdatedAt={dataUpdatedAt} />}
          {data && (
            <span className="text-[10px] font-mono text-muted-foreground">
              {data.count} new · updated {timeAgo(data.fetched_at)} · run_id{" "}
              {data.run_id?.slice(0, 8) ?? "—"}
            </span>
          )}
        </div>
      </div>

      <div
        ref={scrollRef}
        onMouseEnter={pause}
        onMouseLeave={resumeNow}
        onTouchStart={pause}
        onTouchEnd={resumeAfterDelay}
        onWheel={() => {
          pause();
          resumeAfterDelay();
        }}
        className="flex-1 overflow-y-auto p-2 space-y-3"
      >
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
            {g.items.map((item) => {
              const key = newsItemKey(item);
              const isNew = !seenIds.current.has(key);
              return (
              <div
                key={key}
                className={`mb-1.5 p-2 rounded bg-background border border-border ${
                  isNew ? "animate-slide-in-top animate-flash-highlight motion-reduce:animate-none" : ""
                }`}
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
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}
