export type AgentStatus = "Idle" | "Running" | "Complete" | "Skipped-Optional" | "Failed-Fallback";

const STATUS_CLASSES: Record<AgentStatus, string> = {
  Idle: "bg-status-idle/20 text-status-idle border-status-idle/50",
  Running: "bg-status-running/20 text-status-running border-status-running/50",
  Complete: "bg-status-complete/20 text-status-complete border-status-complete/50",
  "Skipped-Optional": "bg-status-skipped/20 text-status-skipped border-status-skipped/50",
  "Failed-Fallback": "bg-status-failed/20 text-status-failed border-status-failed/50",
};

export function AgentNode({
  id,
  name,
  status,
  compact = false,
}: {
  id: string;
  name: string;
  status: AgentStatus;
  compact?: boolean;
}) {
  return (
    <div className="flex flex-col items-center gap-0.5">
      <div
        title={`${id}: ${name} — ${status}`}
        className={`flex items-center justify-center rounded font-mono font-bold text-[10px] border-[1.5px] ${
          compact ? "w-8 h-7" : "w-10 h-9"
        } ${status === "Running" ? "animate-pulse" : ""} ${STATUS_CLASSES[status]}`}
      >
        {id}
      </div>
      {!compact && (
        <span className="text-[9px] text-muted-foreground text-center leading-none" style={{ maxWidth: 44 }}>
          {name}
        </span>
      )}
    </div>
  );
}
