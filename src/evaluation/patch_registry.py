"""
patch_registry.py — Coordinates monkey-patch ownership of shared functions
(currently `call_openai_structured`) between the TruLens wrapper and the
RAGAS tracer so a run that starts both in the same process doesn't
double-patch or restore the wrong original.

Not thread-safe by design: both callers are single-threaded CLI/script
invocations in this codebase's actual usage pattern.
"""

from __future__ import annotations

_active_patches: dict[str, str] = {}


def claim_patch(target: str, owner: str) -> bool:
    """Return True if `owner` now holds the patch on `target`.

    Granted when `target` is unclaimed or already held by `owner`.
    Rejected when a different owner currently holds it.
    """
    current = _active_patches.get(target)
    if current is not None and current != owner:
        return False
    _active_patches[target] = owner
    return True


def release_patch(target: str, owner: str) -> None:
    """Release `owner`'s claim on `target`. No-op if `owner` doesn't hold it."""
    if _active_patches.get(target) == owner:
        del _active_patches[target]
