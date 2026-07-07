# design.md — Supply Chain Command Center

Single source of truth for styling. Every component imports tokens from here (via
`src/styles/tokens.ts` + the Tailwind utility classes defined in `theme.css`) — never a raw
hex value inline in a component file.

## Color tokens

| Token | Hex | Tailwind utility | Existing shadcn equivalent |
|---|---|---|---|
| `bg` | `#0B1220` | `bg-background` | `--background` |
| `panel` | `#121A2B` | `bg-panel` / `bg-card` | `--card` |
| `border` | `#1E293B` | `border-border` | `--border` |
| `border2` | `#243044` | `border-border2` | *(new)* |
| `text` | `#E2E8F0` | `text-foreground` | `--foreground` |
| `textMuted` | `#94A3B8` | `text-muted-strong` | `--secondary-foreground` |
| `textDim` | `#64748B` | `text-muted-foreground` | `--muted-foreground` |

## Risk colors

| Level | Hex | Tailwind utility |
|---|---|---|
| LOW | `#22C55E` | `text-risk-low` / `bg-risk-low` |
| MEDIUM | `#F59E0B` | `text-risk-medium` / `bg-risk-medium` |
| HIGH | `#F97316` | `text-risk-high` / `bg-risk-high` |
| CRITICAL | `#EF4444` | `text-risk-critical` / `bg-risk-critical` |

## Agent status colors

| Status | Hex | Tailwind utility |
|---|---|---|
| Idle | `#334155` | `bg-status-idle` |
| Running | `#F59E0B` | `bg-status-running` |
| Complete | `#22C55E` | `bg-status-complete` |
| Skipped-Optional | `#818CF8` | `bg-status-skipped` |
| Failed-Fallback | `#EF4444` | `bg-status-failed` |

## Typography

| Role | Font | Weight | Size | Usage |
|---|---|---|---|---|
| Display | Inter | 700 | 24px | Screen titles |
| Heading | Inter | 600 | 16px | Panel headers |
| Body | Inter | 400 | 14px | Standard text |
| Caption | Inter | 400 | 12px | Muted labels, timestamps |
| Mono/Numeric | JetBrains Mono | 500 | 13–20px | Scores, percentages, run IDs, log lines |

Loaded via `@fontsource/inter` and `@fontsource/jetbrains-mono` (npm packages, not a CDN link) —
self-contained build, no dependency on `fonts.googleapis.com` at demo time.

Paragraph rules: body `line-height: 1.5`; captions `line-height: 1.3`; `letter-spacing: 0.01em`
on all-caps eyebrow labels (e.g. "GOOGLE NEWS RSS").

## Spacing & radius scale

Tailwind's default spacing scale already maps 1:1 onto the intended scale — use it directly,
don't add a parallel one:

| Name | px | Tailwind step |
|---|---|---|
| xs | 4 | `1` (e.g. `p-1`) |
| sm | 8 | `2` |
| md | 12 | `3` |
| lg | 16 | `4` |
| xl | 24 | `6` |
| 2xl | 32 | `8` |

Radius needs 3 new named values that don't collide with the existing shadcn `--radius-*` scale:

| Name | px | Tailwind utility |
|---|---|---|
| `radius-panel` | 14 | `rounded-panel` |
| `radius-btn` | 10 | `rounded-btn` |
| `radius-pill` | 999 | `rounded-pill` (or built-in `rounded-full`) |

Panels use `rounded-panel`, buttons use `rounded-btn`, badges/pills use `rounded-pill`.

## Elevation & borders

All panels: `1px solid` `border`, no drop shadow — flat "command center" look. Depth comes from
background tone (`panel` vs `bg`), not shadow.

## Animation tokens

```css
@keyframes ping { 75%, 100% { transform: scale(2); opacity: 0; } }
@keyframes pulse { 50% { opacity: 0.5; } }
@keyframes fadeInStagger { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }
```
- `ping` — CRITICAL risk badge indicator dot.
- `pulse` — `Running` agent-node state, live "LIVE" log dot.
- `fadeInStagger` — pipeline-strip node transitions, driven by real backend timing via polling
  (Day 9), not a client-side `setTimeout` chain.

## Component variant contracts

Exact prop shape for every shared component — defined once, never redefined on Days 2–7:

- `<RiskBadge level pulse? size="sm"|"md"|"lg" />`
- `<AgentNode id name status compact? />`
- `<CitationChip source collection />`
- `<Panel title rightContent?>` — standard bordered card wrapper
- `<Button variant="primary"|"ghost"|"disabled">`
- `<Tabs>` — the 6-tab bar; the icon rail is a cosmetic mirror of the same active-tab state
  (single source of truth — do not track two separate `activeTab` states)

## Implementation note (Tailwind v4)

This project uses Tailwind v4 via `@tailwindcss/vite` — there is no `tailwind.config.js`.
Tokens are configured CSS-first: new CSS custom properties go in `:root` in `theme.css`, then
get mirrored into the existing `@theme inline { ... }` block so Tailwind generates the matching
utility classes. See `theme.css` for the exact additions.
