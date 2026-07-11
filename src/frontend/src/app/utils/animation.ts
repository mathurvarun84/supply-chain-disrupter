/**
 * Shared motion-pass helpers (Screens 1/2/6 only — see the task's scope
 * cuts). Deliberately not under hooks/ — that directory is reserved for
 * data-fetching hooks (useLiveFeed, useRiskClassification, useRagEval,
 * etc.), which this animation-only pass must not touch.
 */
import { useEffect, useRef, useState } from "react";

export function prefersReducedMotion(): boolean {
  if (typeof window === "undefined" || !window.matchMedia) return false;
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/**
 * True on first mount, and true again only when `key` changes from its
 * previously-committed value — false on re-renders that carry the same
 * key (e.g. a React Query background refetch that returns identical
 * data). Use `key` as a stable identifier for "this is a genuinely new
 * event" (a run_id, a fetched-once dataset's loaded flag, etc.).
 */
export function useAnimateOnChange(key: unknown): boolean {
  const seen = useRef<{ initialized: boolean; key: unknown }>({ initialized: false, key: undefined });
  const shouldAnimate = !seen.current.initialized || seen.current.key !== key;

  useEffect(() => {
    seen.current = { initialized: true, key };
  }, [key]);

  return shouldAnimate;
}

/**
 * Tweens 0 → target once on mount, and re-tweens whenever `target` changes
 * value (e.g. a new run's composite_score) — but not on a re-render that
 * carries the same target (an identical background refetch), since the
 * effect's dependency array skips those automatically. Jumps straight to
 * target under reduced motion. Returns the raw float — round or
 * `.toFixed()` it yourself at the call site (an integer doc count and a
 * 3-decimal composite score format differently).
 */
export function useCountUp(target: number, durationMs = 500): number {
  const [value, setValue] = useState(() => (prefersReducedMotion() ? target : 0));

  useEffect(() => {
    if (prefersReducedMotion()) {
      setValue(target);
      return;
    }
    let raf = 0;
    const start = performance.now();
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / durationMs);
      setValue(target * t);
      if (t < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target, durationMs]);

  return value;
}
