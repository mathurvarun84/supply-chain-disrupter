/**
 * Types for the Run Pipeline control (top status bar) — POST/GET
 * /api/pipeline/*. Backed by src/api/routers/pipeline.py and
 * src/api/schemas.py (PipelineRunRequest/Response, PipelineStatus).
 */
import type { AgentStatus } from "../components/AgentNode";

export type PipelineMode = "live" | "demo" | "replay";

export type DemoScenarioId =
  | "taiwan_earthquake"
  | "red_sea_crisis"
  | "guardrail_demo"
  | "clean_baseline";

export interface AgentState {
  id: string;
  name: string;
  status: AgentStatus;
  duration_ms: number | null;
}

export interface PipelineStatus {
  run_id: string;
  source_type: "LIVE" | "DEMO-INJECTED" | "REPLAY";
  agents: AgentState[];
  last_ingested_at: string | null;
  openai_status: "connected" | "disconnected";
  langfuse_trace_url: string | null;
  is_complete: boolean;
  // Set only during live mode's pre-L1 connector sweep (before
  // agent_execution_log has any rows for this run_id) — e.g. "Fetching
  // live news & weather data…". Null once L1 starts (the agents[] Running/
  // Complete highlighting takes over from there).
  current_phase: string | null;
}

export interface PipelineRunRequest {
  mode: PipelineMode;
  demo_scenario_id?: DemoScenarioId;
  replay_run_id?: string;
}

export interface PipelineRunResponse {
  run_id: string;
  mode: PipelineMode;
  accepted_at: string;
}

export interface DemoScenarioCard {
  id: DemoScenarioId;
  label: string;
  severity: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
}

// Mirrors src/agents/demo_injector.py's SCENARIO_METADATA — kept in sync by
// hand since there are only 4, fixed scenarios (no /scenarios list endpoint).
export const DEMO_SCENARIOS: DemoScenarioCard[] = [
  { id: "taiwan_earthquake", label: "Taiwan Earthquake", severity: "CRITICAL" },
  { id: "red_sea_crisis", label: "Red Sea Crisis", severity: "HIGH" },
  { id: "guardrail_demo", label: "Prompt-Injection Guardrail Demo", severity: "MEDIUM" },
  { id: "clean_baseline", label: "Clean Baseline", severity: "LOW" },
];
