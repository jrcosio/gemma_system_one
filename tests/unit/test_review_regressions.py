import pytest
from test_hub import FakeHub

from gemma_system_one.doctor import Doctor
from gemma_system_one.hub import CheckpointIntegrityError, download
from gemma_system_one.resources import MemoryTracker


def test_explicit_hash_verification_cannot_succeed_without_reference(cpu_cfg, tmp_path):
    cfg = cpu_cfg.model_copy(update={"model": cpu_cfg.model.model_copy(update={"revision": "a" * 40})})
    hub = FakeHub(tmp_path / "hub", cached=True)

    def offline(_):
        raise ConnectionError("offline")

    with pytest.raises(CheckpointIntegrityError, match="verificar"):
        download(cfg, verify_hash=True, snapshot_fn=hub, metadata_fn=offline)


def test_memory_budget_stops_work_at_sample(monkeypatch):
    monkeypatch.setattr("gemma_system_one.resources.memory_snapshot", lambda: {"process_rss_bytes": 101})
    tracker = MemoryTracker(budget_bytes=100)
    with pytest.raises(RuntimeError, match="Presupuesto"):
        tracker.sample("before_backward")
    assert tracker.peaks()["process_rss_bytes"] == 101


def test_doctor_records_monitor_failure_instead_of_crashing(cpu_cfg, monkeypatch):
    doctor = Doctor(cpu_cfg, skip_model=True)

    def unavailable():
        raise OSError("swap unavailable")

    monkeypatch.setattr("gemma_system_one.resources.memory_snapshot", unavailable)
    called = []
    doctor.step("probe", lambda: called.append(True))
    assert not called
    assert doctor.steps[0]["status"] == "fail"
    assert doctor.steps[0]["error"]["type"] == "OSError"
