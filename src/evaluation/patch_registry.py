"""
patch_registry.py — Coordinates monkey-patch ownership of shared functions
(currently `call_openai_structured`) between the TruLens wrapper and the
RAGAS tracer so a run that starts both in the same process doesn't
double-patch or restore the wrong original.

Reentrant per (target, owner): tracked as (owner, depth) rather than a bare
owner string, so a nested claim/release pair from the SAME owner doesn't
release an outer, still-active claim out from under it — a flat
{target: owner} map would let a completely different owner steal the
target the moment any inner same-owner scope exits, even while an outer
scope of the original owner is still logically active.

Not thread-safe by design: both callers are single-threaded CLI/script
invocations in this codebase's actual usage pattern.
"""

from __future__ import annotations

_active_patches: dict[str, tuple[str, int]] = {}


def claim_patch(target: str, owner: str) -> bool:
    """Return True if `owner` now holds the patch on `target`.

    Granted when `target` is unclaimed (starts a fresh depth-1 claim) or
    already held by `owner` (increments depth — safe to nest). Rejected
    when a different owner currently holds it.
    """
    current = _active_patches.get(target)
    if current is None:
        _active_patches[target] = (owner, 1)
        return True
    current_owner, depth = current
    if current_owner != owner:
        return False
    _active_patches[target] = (owner, depth + 1)
    return True


def release_patch(target: str, owner: str) -> None:
    """Release one level of `owner`'s claim on `target`.

    No-op if `owner` doesn't hold it. Only actually clears the claim once
    depth returns to zero, so an inner (nested) release never releases an
    outer claim still held by the same owner.
    """
    current = _active_patches.get(target)
    if current is None or current[0] != owner:
        return
    _, depth = current
    if depth <= 1:
        del _active_patches[target]
    else:
        _active_patches[target] = (owner, depth - 1)
