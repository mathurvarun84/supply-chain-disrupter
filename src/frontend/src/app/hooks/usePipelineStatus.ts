import { useQuery } from "@tanstack/react-query";
import type { AgentStatus } from "../components/AgentNode";
import { API_BASE_URL } from "../api/config";

export interface AgentState {
  id: string;
  name: string;
  status: AgentStatus;
}

export interface PipelineStatus {
  run_id: string;
  source_type: "LIVE" | "DEMO-INJECTED" | "REPLAY";
  agents: AgentState[];
  last_ingested_at: string | null;
  openai_status: "connected" | "disconnected";
  langfuse_trace_url: string | null;
}

async function fetchPipelineStatus(): Promise<PipelineStatus> {
  const res = await fetch(`${API_BASE_URL}/api/pipeline/status`);
  if (!res.ok) throw new Error(`pipeline/status failed: ${res.status}`);
  return res.json();
}

export function usePipelineStatus() {
  return useQuery({
    queryKey: ["pipeline-status"],
    queryFn: fetchPipelineStatus,
  });
}
