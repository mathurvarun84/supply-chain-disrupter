/**
 * Shared risk-level pill badge for Screen 2. Extracted from the inline
 * RiskBadge in _reference/App.mockup.tsx:195-207 — same markup/classes,
 * now driven by the riskColors Tailwind tokens (text-risk-N, bg-risk-N)
 * instead of inline hex, matching Screen 1's panel convention.
 */
import type { RiskLevel } from "../../types/riskClassification";

const RISK_CLASS: Record<RiskLevel, string> = {
  LOW: "text-risk-low bg-risk-low/10 border-risk-low/25",
  MEDIUM: "text-risk-medium bg-risk-medium/10 border-risk-medium/25",
  HIGH: "text-risk-high bg-risk-high/10 border-risk-high/25",
  CRITICAL: "text-risk-critical bg-risk-critical/10 border-risk-critical/25",
};

const SIZE_CLASS = {
  sm: "text-[10px] px-2 py-0.5",
  md: "text-xs px-2.5 py-1",
  lg: "text-xl px-5 py-2 font-bold",
};

export function RiskBadge({
  level,
  pulse = false,
  size = "md",
}: {
  level: RiskLevel | "N/A" | null;
  pulse?: boolean;
  size?: "sm" | "md" | "lg";
}) {
  if (level === null || level === "N/A") {
    return (
      <span className={`inline-flex items-center gap-1.5 rounded font-mono font-semibold tracking-widest shrink-0 border border-border text-muted-foreground ${SIZE_CLASS[size]}`}>
        N/A
      </span>
    );
  }
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded font-mono font-semibold tracking-widest shrink-0 border ${RISK_CLASS[level]} ${SIZE_CLASS[size]}`}
    >
      {pulse && level === "CRITICAL" && (
        <span className="inline-block w-1.5 h-1.5 rounded-full bg-risk-critical animate-ping" />
      )}
      {level}
    </span>
  );
}
