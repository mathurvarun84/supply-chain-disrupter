/**
 * Screen 2 (Risk Classification) tab body — assembles the verdict card,
 * three-signal ensemble row, Judge panel, and Escalation Guard panel,
 * matching the layout from _reference/App.mockup.tsx's
 * TabRiskClassification. Wires to real ensemble data via GET
 * /api/risk-classification/{run_id} (see useRiskClassification.ts), which
 * reads the risk_classification_output snapshot written as soon as L4
 * finishes for the active run — not just "whatever was last classified."
 * Falls back to /latest only when no run_id is active yet. This component
 * only owns layout + loading/error states, not data fetching.
 */
import { useRiskClassification } from "./hooks/useRiskClassification";
import { VerdictCard } from "./components/risk-classification/VerdictCard";
import { SignalEnsembleRow } from "./components/risk-classification/SignalEnsembleRow";
import { JudgePanel } from "./components/risk-classification/JudgePanel";
import { EscalationGuardPanel } from "./components/risk-classification/EscalationGuardPanel";

export function TabRiskClassification({ runId }: { runId?: string }) {
  const { data, isLoading, isError } = useRiskClassification(runId);

  if (isLoading) {
    return (
      <div className="h-full flex items-center justify-center text-xs text-muted-foreground">
        Loading risk classification…
      </div>
    );
  }

  if (isError || !data) {
    // While a run is active, a 404 just means L4 hasn't finished yet —
    // useRiskClassification keeps polling every 2s, so this is a waiting
    // state, not a hard failure. Only show the harsh error when there's no
    // active run to wait on (the /latest fallback itself came back empty).
    if (runId) {
      return (
        <div className="h-full flex items-center justify-center text-xs text-muted-foreground">
          Waiting for L4 risk classification to finish…
        </div>
      );
    }
    return (
      <div className="h-full flex items-center justify-center text-xs text-risk-critical">
        Could not load risk classification. Check backend logs.
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto p-3 space-y-3">
      <VerdictCard data={data} />

      <div className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider px-1">
        Three-Signal Ensemble — Independent Witnesses
      </div>
      <SignalEnsembleRow data={data} />

      <JudgePanel data={data} />
      <EscalationGuardPanel />
    </div>
  );
}
