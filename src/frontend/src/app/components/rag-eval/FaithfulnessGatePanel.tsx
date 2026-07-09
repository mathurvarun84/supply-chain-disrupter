/**
 * Screen 6 Faithfulness Gate explainer — static threshold rule plus the
 * passing/failing example pair. Frontend half of the same gate Screen
 * 5's Guardrails sub-tab shows firing (GUARDRAIL_TABLE's
 * "faithfulness-gate" row, src/api/fixtures.py) — the 0.75 threshold and
 * 0.87/0.61 example pair must not drift from that fixture. Not wired to
 * a live query; ported from _reference/App.mockup.tsx:1011-1027.
 */
import { useEffect, useState } from "react";

const FAITHFULNESS_THRESHOLD = 0.75;

export function FaithfulnessGatePanel() {
  // One-shot glow on the passing example: render with the glow "on", then
  // let a CSS transition fade it out over the next tick — avoids adding a
  // 4th/5th bespoke keyframe for a single fade-out.
  const [glowed, setGlowed] = useState(false);
  useEffect(() => {
    const raf = requestAnimationFrame(() => setGlowed(true));
    return () => cancelAnimationFrame(raf);
  }, []);

  return (
    <div className="rounded-lg p-4 bg-card border border-border">
      <div className="text-xs font-semibold text-muted-strong mb-2">Faithfulness Gate Status</div>
      <div className="text-[11px] text-muted-foreground leading-relaxed mb-3">
        When <span className="font-mono text-risk-medium">faithfulness &lt; {FAITHFULNESS_THRESHOLD}</span> →
        mitigation plan routed to human review, Slack suppressed.
      </div>
      <div className="grid grid-cols-2 gap-2">
        <div
          className={`p-2 rounded text-center text-[10px] bg-risk-low/5 border border-risk-low/30 transition-shadow duration-[1200ms] ease-out motion-reduce:transition-none ${
            glowed ? "shadow-none" : "shadow-[0_0_16px_rgba(34,197,94,0.35)]"
          }`}
        >
          <div className="font-mono font-bold text-risk-low">0.87 ✓</div>
          <div className="text-muted-foreground mt-0.5">Current — Slack allowed</div>
        </div>
        <div className="p-2 rounded text-center text-[10px] bg-risk-critical/5 border border-risk-critical/30 animate-pulse motion-reduce:animate-none">
          <div className="font-mono font-bold text-risk-critical">0.61 ✗</div>
          <div className="text-muted-foreground mt-0.5">Example — human review</div>
        </div>
      </div>
    </div>
  );
}
