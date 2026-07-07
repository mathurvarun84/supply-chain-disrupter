export function TabPlaceholder({ title, day }: { title: string; day: number }) {
  return (
    <div className="h-full flex items-center justify-center">
      <div className="text-center">
        <div className="text-sm font-semibold text-foreground mb-1">{title}</div>
        <div className="text-xs font-mono text-muted-foreground">Coming Day {day}</div>
      </div>
    </div>
  );
}
