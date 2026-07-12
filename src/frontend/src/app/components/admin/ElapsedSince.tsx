/**
 * Ticks "Ns elapsed" once per second from an ISO timestamp — used by the
 * Admin build cards so a running job visibly progresses between polls
 * instead of sitting on a static "Building…" label.
 */
import { useEffect, useState } from "react";

export function ElapsedSince({ isoTimestamp }: { isoTimestamp: string }) {
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1_000);
    return () => clearInterval(id);
  }, []);

  const elapsedS = Math.max(0, Math.round((now - new Date(isoTimestamp).getTime()) / 1000));
  return <span>{elapsedS}s elapsed</span>;
}
