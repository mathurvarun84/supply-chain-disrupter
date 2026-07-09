/**
 * Screen 2 static guardrail explanation panel. Does NOT bind to per-run
 * data — it's the evaluator-facing explanation of the escalation guard
 * design, text reproduced from docs/ARCHITECTURE.md's "Final Label
 * Fallback Chain" section (lines 397-406), not a paraphrase. Layout
 * ported from _reference/App.mockup.tsx:500-509.
 */
import { Lock } from "lucide-react";

export function EscalationGuardPanel() {
  return (
    <div className="rounded-lg p-4 bg-card border border-risk-critical/35">
      <div className="flex items-center gap-2 mb-2">
        <Lock size={13} className="text-risk-critical shrink-0" />
        <span className="text-xs font-semibold text-risk-critical">
          Escalation Guard — Code-Enforced, Not LLM-Trusted
        </span>
      </div>
      <div className="text-[11px] text-muted-foreground leading-relaxed font-mono">
        Final label fallback chain: judge_verdict.final_label →{" "}
        llm_signal.predicted_label (if judge unavailable) →{" "}
        rule_signal.escalated_label (if LLM unavailable). "Shipping canceled"
        always forces CRITICAL regardless of judge output.{" "}
        <span className="text-status-running">critical_flag = True</span> →{" "}
        <span className="text-risk-critical">slack_should_fire = True</span> is
        set only when final_label == "CRITICAL" — never from the judge verdict
        alone, and never trusted from any LLM response field directly.
      </div>
    </div>
  );
}
