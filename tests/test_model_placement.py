from types import SimpleNamespace

import pytest

from alignment_stress_lab.model_placement import require_no_cpu_or_disk_offload


def test_all_gpu_placement_is_accepted() -> None:
    require_no_cpu_or_disk_offload(SimpleNamespace(hf_device_map={"a": 0, "b": "cuda:0"}))


@pytest.mark.parametrize("device", ["cpu", "disk"])
def test_offloaded_placement_is_rejected(device: str) -> None:
    model = SimpleNamespace(hf_device_map={"a": 0, "b": device})
    with pytest.raises(RuntimeError, match="CPU/disk-offloaded"):
        require_no_cpu_or_disk_offload(model)
