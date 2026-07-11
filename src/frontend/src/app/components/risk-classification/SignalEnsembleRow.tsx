/**
 * Screen 2 three-signal grid — Rule-Based / DistilBERT / GPT-4o+RAG,
 * mirroring Screen 1's News/Weather/Log three-column layout. Data
 * source: RiskClassification.rule_signal / .distilbert_signal /
 * .llm_signal. Each column renders its own "signal unavailable" state
 * independently — DistilBERT when model_source isn't "fine-tuned", LLM
 * when predicted_label is null (OPENAI_API_KEY unset or the call
 * failed) — this is new relative to the mockup, which only shows the
 * happy path, but is required by the real ensemble's partial-
 * availability contract. Layout ported from
 * _reference/App.mockup.tsx:415-487.
 */
import { useEffect, useState } from "react";
import { ChevronDown } from "lucide-react";
import type { RiskClassification } from "../../types/riskClassification";
import { RiskBadge } from "./RiskBadge";
import { CitationChip } from "./CitationChip";
import { useAnimateOnChange, useCountUp } from "../../utils/animation";

const STAGGER_MS = 90;
function staggerAnim(shouldAnimate: boolean, index: number) {
  return {
    className: shouldAnimate ? " animate-fade-stagger motion-reduce:animate-none" : "",
    style: shouldAnimate ? { animationDelay: `${index * STAGGER_MS}ms` } : undefined,
  };
}

const COMPONENT_WEIGHTS: { key: keyof RiskClassification["rule_signal"]; label: string; w: number }[] = [
  { key: "geo_component", label: "geo", w: 0.4 },
  { key: "supply_component", label: "supply_disruption", w: 0.3 },
  { key: "freight_component", label: "freight", w: 0.15 },
  { key: "defect_component", label: "defect", w: 0.15 },
];

// Each row rolls its weighted value up from 0 and fills its bar from 0% —
// pulled out of the .map() below because hooks (useCountUp) can't be
// called inside a loop body directly.
function RuleComponentRow({
  label,
  weight,
  value,
  grown,
}: {
  label: string;
  weight: number;
  value: number;
  grown: boolean;
}) {
  const displayWeighted = useCountUp(value * weight, 600);
  return (
    <div className="mb-2">
      <div className="flex justify-between text-[10px] mb-1">
        <span className="font-mono text-muted-foreground">{label} ×{weight}</span>
        <span className="font-mono text-foreground">{displayWeighted.toFixed(3)}</span>
      </div>
      <div className="h-1 rounded-full overflow-hidden bg-border">
        <div
          className="h-full rounded-full bg-primary transition-[width] duration-[600ms] ease-out motion-reduce:transition-none"
          style={{ width: grown ? `${Math.min(value, 1) * 100}%` : "0%" }}
        />
      </div>
    </div>
  );
}

export function SignalEnsembleRow({ data }: { data: RiskClassification }) {
  const [expanded, setExpanded] = useState(false);
  const { rule_signal, distilbert_signal, llm_signal } = data;
  const shouldAnimate = useAnimateOnChange(data.run_id);

  // Bars start at 0 width and grow to their real value one frame after
  // mount (same two-phase trick as Screen 6's scorecard bars) — only on a
  // genuinely new run, never replayed on an identical re-render.
  const [grown, setGrown] = useState(!shouldAnimate);
  useEffect(() => {
    if (!shouldAnimate) return;
    setGrown(false);
    const raf = requestAnimationFrame(() => setGrown(true));
    return () => cancelAnimationFrame(raf);
  }, [shouldAnimate, data.run_id]);

  const displayComposite = useCountUp(rule_signal.composite_score, 600);
  const displayConfidencePct = useCountUp((distilbert_signal.confidence ?? 0) * 100, 600);

  return (
    <div className="grid grid-cols-3 gap-3">
      {/* Rule-Based */}
      <div
        className={`rounded-lg p-4 bg-card border border-border${staggerAnim(shouldAnimate, 0).className}`}
        style={staggerAnim(shouldAnimate, 0).style}
      >
        <div className="text-xs font-semibold text-primary mb-3">Rule-Based Composite</div>
        {COMPONENT_WEIGHTS.map((c) => (
          <RuleComponentRow
            key={c.label}
            label={c.label}
            weight={c.w}
            value={rule_signal[c.key] as number}
            grown={grown}
          />
        ))}
        <div className="mt-3 pt-3 flex justify-between items-center border-t border-border">
          <span className="text-[10px] font-mono text-muted-foreground">
            composite = {displayComposite.toFixed(3)}
          </span>
          <RiskBadge level={rule_signal.escalated_label} size="sm" />
        </div>
      </div>

      {/* DistilBERT */}
      <div
        className={`rounded-lg p-4 flex flex-col bg-card border border-border${staggerAnim(shouldAnimate, 1).className}`}
        style={staggerAnim(shouldAnimate, 1).style}
      >
        <div className="text-xs font-semibold text-accent mb-3">Fine-Tuned DistilBERT</div>
        {distilbert_signal.model_source !== "fine-tuned" ? (
          <div className="flex-1 flex items-center justify-center text-center text-[11px] text-muted-foreground py-4">
            Signal unavailable — model not loaded ({distilbert_signal.model_source})
          </div>
        ) : (
          <>
            <div className="flex flex-col items-center flex-1 justify-center py-2">
              <RiskBadge level={distilbert_signal.predicted_label} />
              <div className="mt-3 text-center">
                <div className="text-4xl font-mono font-bold text-foreground">
                  {distilbert_signal.confidence !== null ? `${Math.round(displayConfidencePct)}%` : "—"}
                </div>
                <div className="text-[10px] text-muted-foreground">confidence</div>
              </div>
            </div>
            <div className="h-1.5 rounded-full overflow-hidden mb-3 bg-border">
              <div
                className="h-full rounded-full bg-accent transition-[width] duration-[600ms] ease-out motion-reduce:transition-none"
                style={{ width: grown ? `${(distilbert_signal.confidence ?? 0) * 100}%` : "0%" }}
              />
            </div>
            <div className="text-[10px] font-mono text-muted-foreground leading-relaxed">
              66M params · local inference
              <br />
              {distilbert_signal.inference_ms !== null ? `${distilbert_signal.inference_ms.toFixed(0)}ms` : "—"} · temperature=0.0
            </div>
          </>
        )}
      </div>

      {/* GPT-4o + RAG */}
      <div
        className={`rounded-lg p-4 bg-card border border-border${staggerAnim(shouldAnimate, 2).className}`}
        style={staggerAnim(shouldAnimate, 2).style}
      >
        <div className="text-xs font-semibold text-risk-low mb-3">GPT-4o + RAG</div>
        {llm_signal.predicted_label === null ? (
          <div className="flex items-center justify-center text-center text-[11px] text-muted-foreground py-8">
            Signal unavailable — OPENAI_API_KEY not set or the call failed
          </div>
        ) : (
          <>
            <div className="flex items-center gap-2 mb-3">
              <RiskBadge level={llm_signal.predicted_label} size="sm" />
              <span className="text-[10px] font-mono text-muted-foreground">
                {llm_signal.primary_driver ?? "—"} driver · {llm_signal.confidence_level ?? "—"} confidence
              </span>
            </div>
            <button
              onClick={() => setExpanded(!expanded)}
              className="w-full text-left text-[11px] p-2 rounded mb-3 transition-colors bg-background border border-border"
            >
              <div className="flex items-center justify-between text-muted-foreground">
                <span className="font-semibold">Justification</span>
                <ChevronDown size={11} className={`transition-transform ${expanded ? "rotate-180" : ""}`} />
              </div>
              {expanded && (
                <div className="mt-2 text-muted-foreground leading-relaxed">
                  "{llm_signal.rationale}"
                </div>
              )}
            </button>
            <div className="flex flex-wrap gap-1">
              {llm_signal.rag_citations.length === 0 ? (
                <span className="text-[10px] text-muted-foreground">No citations retrieved.</span>
              ) : (
                llm_signal.rag_citations.map((c) => <CitationChip key={c} source={c} />)
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
