/**
 * Screen 1 (Live Feed) tab body — assembles the News/Weather/Agent-Log
 * 3-column grid plus the Gantt strip, matching the layout from
 * _reference/App.mockup.tsx's TabLiveFeed. All child panels wire to real
 * L1 data via GET /api/live-feed/* (see each panel's own header comment);
 * this component only owns layout, not data fetching.
 */
import { RefreshControl } from "./components/live-feed/RefreshControl";
import { NewsPanel } from "./components/live-feed/NewsPanel";
import { WeatherPanel } from "./components/live-feed/WeatherPanel";
import { AgentLogPanel } from "./components/live-feed/AgentLogPanel";
import { GanttStrip } from "./components/live-feed/GanttStrip";

export function TabLiveFeed({ onTabSwitch }: { onTabSwitch: (t: number) => void }) {
  return (
    <div className="flex flex-col h-full overflow-hidden">
      <RefreshControl />
      <div
        className="flex-1 grid gap-3 p-3 overflow-hidden"
        style={{ gridTemplateColumns: "1fr 1fr 1fr" }}
      >
        <NewsPanel />
        <WeatherPanel />
        <AgentLogPanel onTabSwitch={onTabSwitch} />
      </div>
      <GanttStrip />
    </div>
  );
}
