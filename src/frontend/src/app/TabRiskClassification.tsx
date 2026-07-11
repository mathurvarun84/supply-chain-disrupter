/**
 * Screen 2 (Risk Classification) tab body — assembles the verdict card,
 * three-signal ensemble row, Judge panel, and Escalation Guard panel,
 * matching the layout from _reference/App.mockup.tsx's
 * TabRiskClassification. Wires to real ensemble data via GET
 * /api/risk-classification/latest (see useRiskClassification.ts); this
 * component only owns layout + loading/error states, not data fetching.
 */
import { useRiskClassification } from "./hooks/useRiskClassification";
import { VerdictCard } from "./components/risk-classification/VerdictCard";
import { SignalEnsembleRow } from "./components/risk-classification/SignalEnsembleRow";
import { JudgePanel } from "./components/risk-classification/JudgePanel";
import { EscalationGuardPanel } from "./components/risk-classification/EscalationGuardPanel";

export function TabRiskClassification() {
  const { data, isLoading, isError } = useRiskClassification();

  if (isLoading) {
    return (
      <div className="h-full flex items-center justify-center text-xs text-muted-foreground">
        Loading risk classification…
      </div>
    );
  }

  if (isError || !data) {
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
