/**
 * Screen 2 top card — final label badge + radial SVG gauge of
 * composite_score against `threshold`. Data source:
 * RiskClassification.final_label / .rule_signal.composite_score /
 * .threshold / .mode / .from_cache. Consumed by TabRiskClassification.tsx.
 * Layout/markup ported from _reference/App.mockup.tsx:384-413 (inline SVG
 * gauge, not a chart-library dependency for a single ring).
 */
import { useEffect, useRef, useState } from "react";
import type { RiskClassification } from "../../types/riskClassification";
import { RiskBadge } from "./RiskBadge";
import { useAnimateOnChange, useCountUp } from "../../utils/animation";

const CIRCUMFERENCE = 2 * Math.PI * 38; // r=38, matches the mockup's radius

// NOTE: this gauge is hand-rolled SVG (per the header comment above it was
// ported straight from the mockup's inline-SVG gauge), not Recharts'
// RadialBarChart as design.md's original component contract implies for
// Screen 2 — a pre-existing deviation, not introduced by this pass. That
// ruled out the "prefer Recharts' isAnimationActive prop" route from Step 4
// of the animation spec; the fill is animated here via a CSS transition on
// stroke-dashoffset instead (the countFill token's mechanism, applied
// through a transition rather than a literal @keyframes countFill).
export function VerdictCard({ data }: { data: RiskClassification }) {
  const { final_label, rule_signal, threshold, run_id, mode, from_cache } = data;
  const score = rule_signal.composite_score;
  const targetOffset = CIRCUMFERENCE * (1 - Math.min(score, 1));

  const shouldAnimateGauge = useAnimateOnChange(run_id);
  // Mount the ring at "empty" for one frame, then let the CSS transition on
  // stroke-dashoffset carry it to targetOffset — the standard two-phase
  // trick for animating a value that has to start at 0 on a genuinely new
  // run but must never replay on an identical re-render.
  const [ringOffset, setRingOffset] = useState(shouldAnimateGauge ? CIRCUMFERENCE : targetOffset);
  useEffect(() => {
    if (!shouldAnimateGauge) {
      setRingOffset(targetOffset);
      return;
    }
    setRingOffset(CIRCUMFERENCE);
    const raf = requestAnimationFrame(() => setRingOffset(targetOffset));
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [run_id, targetOffset]);

  const prevScore = useRef<number | null>(null);
  const [scoreFlash, setScoreFlash] = useState(false);
  useEffect(() => {
    if (prevScore.current !== null && prevScore.current !== score) {
      setScoreFlash(true);
      const t = setTimeout(() => setScoreFlash(false), 1200);
      prevScore.current = score;
      return () => clearTimeout(t);
    }
    prevScore.current = score;
  }, [score]);

  // Same 800ms window as the ring fill so the number "arrives" in step
  // with the gauge, not after it. Re-tweens only when score itself
  // changes value (see useCountUp) — never on an identical re-render.
  const displayScore = useCountUp(score, 800);

  return (
    <div className="rounded-panel p-5 bg-card border border-border">
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <div>
          <div className="text-[10px] font-mono text-muted-foreground uppercase tracking-widest mb-2">
            Risk Verdict — run_id {run_id} ({mode}
            {from_cache ? ", cached" : ""})
          </div>
          <RiskBadge level={final_label} pulse size="lg" />
          <div
            className={`mt-3 flex items-center gap-2 flex-wrap rounded ${
              scoreFlash ? "animate-flash-highlight motion-reduce:animate-none" : ""
            }`}
          >
            <span className="text-[10px] font-mono text-muted-foreground">
              composite_score: <strong className="text-foreground">{displayScore.toFixed(3)}</strong>
            </span>
            <span className="text-[10px] font-mono text-muted-foreground">
              threshold: {threshold.toFixed(2)}
            </span>
          </div>
        </div>

        <div className="flex flex-col items-center shrink-0">
          <div className="relative w-24 h-24">
            <svg viewBox="0 0 100 100" className="w-full h-full -rotate-90">
              <circle cx="50" cy="50" r="38" fill="none" stroke="var(--border)" strokeWidth="9" />
              <circle
                cx="50" cy="50" r="38" fill="none"
                stroke="var(--risk-critical)" strokeWidth="9"
                strokeDasharray={CIRCUMFERENCE}
                strokeDashoffset={ringOffset}
                strokeLinecap="round"
                className="transition-[stroke-dashoffset] duration-[800ms] ease-out motion-reduce:transition-none"
              />
              <circle
                cx="50" cy="50" r="38" fill="none"
                stroke="var(--status-running)" strokeWidth="3"
                strokeDasharray={`2 ${CIRCUMFERENCE - 2}`}
                strokeDashoffset={`-${threshold * CIRCUMFERENCE}`}
                strokeLinecap="round"
              />
            </svg>
            <div className="absolute inset-0 flex flex-col items-center justify-center">
              <span className="text-lg font-mono font-bold text-foreground">{displayScore.toFixed(3)}</span>
              <span className="text-[9px] text-muted-foreground">composite</span>
            </div>
          </div>
          <span className="text-[9px] font-mono text-status-running">↑ threshold {threshold.toFixed(2)}</span>
        </div>
      </div>
    </div>
  );
}
