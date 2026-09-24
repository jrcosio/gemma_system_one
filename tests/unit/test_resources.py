from gemma_system_one.resources import GIB, MemoryTracker, memory_snapshot


def test_snapshot_has_process_and_system_fields():
    snap = memory_snapshot()
    for key in ("process_rss_bytes", "system_total_bytes", "swap_used_bytes", "vm_pressure_level"):
        assert key in snap
    assert snap["process_rss_bytes"] > 0


def test_peaks_and_budget_are_per_metric_not_summed():
    tracker = MemoryTracker(budget_bytes=10 * GIB)
    tracker.samples = [
        {
            "label": "a",
            "process_rss_bytes": 6 * GIB,
            "mps_driver_allocated_bytes": 6 * GIB,
            "swap_used_bytes": 0,
        },
        {
            "label": "b",
            "process_rss_bytes": 2 * GIB,
            "mps_driver_allocated_bytes": 7 * GIB,
            "swap_used_bytes": GIB,
        },
    ]
    assert tracker.peaks()["mps_driver_allocated_bytes"] == 7 * GIB
    # 6 + 7 > 10, pero RSS y driver pueden solaparse en memoria unificada: no se suman.
    assert tracker.over_budget() == []
    assert tracker.summary()["swap_delta_bytes"] == GIB
    tracker.samples.append({"label": "c", "mps_driver_allocated_bytes": 11 * GIB})
    assert tracker.over_budget() == ["mps_driver_allocated_bytes"]
