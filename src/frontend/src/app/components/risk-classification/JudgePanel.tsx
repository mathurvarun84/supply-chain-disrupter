/**
 * Screen 2 LLM-as-Judge arbitration panel. Data source:
 * RiskClassification.judge_verdict — surfaces .reasoning and, when
 * signals_agreed is false, .disagreement_explanation, which is the
 * evaluator "wow moment" the ensemble was built to showcase (see
 * docs/ARCHITECTURE.md). When judge_verdict is null (Judge call failed
 * or was skipped — no OPENAI_API_KEY), renders an explicit fallback
 * explaining final_label came from llm_signal/rule_signal instead,
 * rather than hiding the gap. Layout ported from
 * _reference/App.mockup.tsx:490-498.
 */
import type { RiskClassification } from "../../types/riskClassification";
import { useAnimateOnChange } from "../../utils/animation";

export function JudgePanel({ data }: { data: RiskClassification }) {
  const { judge_verdict, final_label, llm_signal, rule_signal } = data;
  const shouldAnimate = useAnimateOnChange(data.run_id);
  const mountClass = shouldAnimate ? " animate-fade-stagger motion-reduce:animate-none" : "";

  if (judge_verdict === null) {
    const fallbackSource = llm_signal.predicted_label !== null ? "GPT-4o Signal 3" : "the rule-based composite";
    return (
      <div className={`rounded-lg p-4 bg-card border border-border${mountClass}`}>
        <div className="flex items-center gap-3 mb-2">
          <span className="text-xs font-semibold text-risk-medium">LLM-as-Judge Arbitration</span>
          <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-risk-medium/10 text-risk-medium">
            JUDGE UNAVAILABLE
          </span>
        </div>
        <div className="text-[11px] font-mono text-muted-foreground leading-relaxed">
          Judge call failed or OPENAI_API_KEY is unset — final_label ({final_label}) fell back to {fallbackSource}
          {rule_signal.delivery_status_override ? ` (delivery_status override: ${rule_signal.delivery_status_override})` : ""}.
        </div>
      </div>
    );
  }

  const consensusLabel = judge_verdict.signals_agreed ? "CONSENSUS — no arbitration needed" : "ARBITRATED — signals disagreed";
  const consensusClass = judge_verdict.signals_agreed
    ? "bg-risk-low/10 text-risk-low"
    : "bg-risk-medium/10 text-risk-medium";

  return (
    <div className={`rounded-lg p-4 bg-card border border-border${mountClass}`}>
      <div className="flex items-center gap-3 mb-2 flex-wrap">
        <span className="text-xs font-semibold text-risk-medium">LLM-as-Judge Arbitration</span>
        <span className={`text-[10px] font-mono px-2 py-0.5 rounded ${consensusClass}`}>{consensusLabel}</span>
        <span className="text-[10px] font-mono text-muted-foreground">verdict_type: {judge_verdict.verdict_type}</span>
      </div>
      <div className="text-[11px] font-mono text-muted-foreground leading-relaxed">
        "{judge_verdict.reasoning}"
      </div>
      {!judge_verdict.signals_agreed && judge_verdict.disagreement_explanation && (
        <div className="mt-2 p-2 rounded bg-risk-medium/10 border border-risk-medium/25 text-[11px] text-foreground leading-relaxed">
          <span className="font-semibold text-risk-medium">Disagreement: </span>
          {judge_verdict.disagreement_explanation}
        </div>
      )}
    </div>
  );
}
