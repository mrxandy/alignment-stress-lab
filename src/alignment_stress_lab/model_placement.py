"""Fail fast when a model intended for GPU execution spills to CPU or disk."""

from __future__ import annotations

from typing import Any


def require_no_cpu_or_disk_offload(model: Any) -> None:
    """Reject Accelerate auto-placement that would make GPU tests misleading."""

    device_map = getattr(model, "hf_device_map", None)
    if not isinstance(device_map, dict):
        return
    offloaded = [
        name
        for name, device in device_map.items()
        if str(device).lower() in ("cpu", "disk")
    ]
    if offloaded:
        raise RuntimeError(
            f"Model has {len(offloaded)} CPU/disk-offloaded modules; "
            "free GPU memory before retrying (no generation started)"
        )
