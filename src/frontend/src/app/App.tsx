import { useState } from "react";
import {
  Activity, BarChart2, Shield, Eye, Database, Settings, Play,
  Copy, Clock, Server, Map,
} from "lucide-react";
import { AgentNode } from "./components/AgentNode";
import { TabPlaceholder } from "./components/TabPlaceholder";
import { TabLiveFeed } from "./TabLiveFeed";
import { TabRiskClassification } from "./TabRiskClassification";
import { TabRagEval } from "./TabRagEval";
import { usePipelineStatus } from "./hooks/usePipelineStatus";

const TABS = [
  { icon: Activity, label: "Live Feed", day: 2 },
  { icon: Shield, label: "Risk Classification", day: 3 },
  { icon: BarChart2, label: "Forecast & Simulation", day: 4 },
  { icon: Map, label: "Mitigation Plan", day: 5 },
  { icon: Eye, label: "Observability", day: 6 },
  { icon: Database, label: "RAG / RAGAS", day: 7 },
];

export default function App() {
  const [activeTab, setActiveTab] = useState(0);
  const { data: pipeline } = usePipelineStatus();

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-background text-foreground" style={{ fontFamily: "'Inter', sans-serif" }}>
      {/* Top Status Bar */}
      <div className="flex items-center gap-4 px-4 py-2 shrink-0 bg-panel border-b border-border">
        {/* Wordmark */}
        <div className="flex items-center gap-2 shrink-0 mr-1">
          <div className="w-6 h-6 rounded flex items-center justify-center bg-gradient-to-br from-primary to-accent">
            <Activity size={11} className="text-white" />
          </div>
          <div>
            <div className="text-[11px] font-bold text-foreground leading-none">Supply Chain</div>
            <div className="text-[9px] text-muted-foreground leading-none tracking-wide">Command Center</div>
          </div>
        </div>

        {/* Pipeline Strip */}
        <div className="flex items-center gap-1 flex-1 justify-center min-w-0">
          {(pipeline?.agents ?? []).map((agent, i, arr) => (
            <div key={agent.id} className="flex items-center gap-1">
              <AgentNode id={agent.id} name={agent.name} status={agent.status} compact />
              {i < arr.length - 1 && <div className="w-4 h-px bg-border" />}
            </div>
          ))}
        </div>

        {/* Right Controls */}
        <div className="flex items-center gap-3 shrink-0">
          <div className="flex items-center gap-1 cursor-pointer group" title="Click to copy run_id">
            <span className="text-[9px] font-mono text-muted-foreground">run_id</span>
            <span className="text-[9px] font-mono text-muted-strong group-hover:text-foreground transition-colors">
              {pipeline?.run_id ?? "—"}
            </span>
            <Copy size={8} className="text-muted-foreground group-hover:text-muted-strong" />
          </div>

          <span className="text-[9px] font-mono px-2 py-0.5 rounded-pill flex items-center gap-1 bg-secondary text-muted-strong border border-border">
            {pipeline?.source_type ?? "REPLAY"}
          </span>

          <div className="flex items-center gap-1 text-[9px] font-mono text-muted-foreground">
            <Clock size={9} />
            <span>{pipeline?.last_ingested_at ?? "never"}</span>
          </div>

          <div className="text-[9px] font-mono flex items-center gap-1 px-1.5 py-0.5 rounded text-risk-low bg-risk-low/10 border border-risk-low/25">
            <Server size={9} />OPENAI: {pipeline?.openai_status ?? "connected"}
          </div>

          <button
            disabled
            title="Wired on Day 9"
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-btn text-xs font-semibold text-white opacity-50 cursor-not-allowed bg-secondary border border-primary/20"
          >
            <Play size={11} fill="white" />
            Run Pipeline
          </button>
        </div>
      </div>

      {/* Main Layout */}
      <div className="flex flex-1 overflow-hidden">
        {/* Icon Rail */}
        <div className="flex flex-col items-center gap-1 py-3 shrink-0 w-12 bg-panel border-r border-border">
          {TABS.map((tab, i) => (
            <button
              key={i}
              onClick={() => setActiveTab(i)}
              title={tab.label}
              className={`flex items-center justify-center w-9 h-9 rounded transition-all ${
                activeTab === i ? "bg-primary/10 text-primary" : "text-status-idle"
              }`}
            >
              <tab.icon size={15} />
            </button>
          ))}
          <div className="mt-auto">
            <button title="Settings" className="flex items-center justify-center w-9 h-9 rounded text-status-idle transition-colors hover:text-muted-foreground">
              <Settings size={15} />
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Tab Bar */}
          <div className="flex items-center gap-0.5 px-3 pt-2 shrink-0 bg-panel border-b border-border">
            {TABS.map((tab, i) => (
              <button
                key={i}
                onClick={() => setActiveTab(i)}
                className={`flex items-center gap-1.5 px-3 pb-2 text-[11px] font-medium transition-colors border-b-2 ${
                  activeTab === i ? "text-primary border-primary" : "text-muted-foreground border-transparent"
                }`}
              >
                <tab.icon size={11} />
                {tab.label}
              </button>
            ))}
          </div>

          {/* Tab Body */}
          <div className="flex-1 overflow-hidden bg-background">
            {activeTab === 0 ? (
              <TabLiveFeed onTabSwitch={setActiveTab} />
            ) : activeTab === 1 ? (
              <TabRiskClassification />
            ) : activeTab === 5 ? (
              <TabRagEval />
            ) : (
              <TabPlaceholder title={TABS[activeTab].label} day={TABS[activeTab].day} />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
