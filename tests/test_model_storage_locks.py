"""Managed preparations participate even on a cache hit or before activation."""

import pytest

from sparkrun.core import model_storage_locks as locks
from sparkrun.models.storage_remote import query, WorkerError
from sparkrun.models import storage_worker as worker


def test_conflicting_repository_preparation_refused(tmp_path):
    home = str(tmp_path / "hf")
    repos = [{"repo_id": "org/model"}]
    query("localhost", home, "reserve", operation="one", repos=repos)
    with pytest.raises(WorkerError, match="Another preparation"):
        query("localhost", home, "reserve", operation="two", repos=repos)
    # A different repository can still be prepared concurrently.
    query("localhost", home, "reserve", operation="two", repos=[{"repo_id": "org/other"}])
    assert len(worker.read_records(tmp_path / "hf/hub", "reservations")) == 2


def test_nested_guards_keep_outer_reservation_until_activation_returns(tmp_path):
    home = str(tmp_path / "hf")
    locations = [("localhost", home, {})]
    repos = [{"repo_id": "org/model", "revision": None}]

    @locks.guarded_operation
    def inner():
        locks._operation_stack.get().enter_context(locks.protect_caches(locations, repos))
        records = worker.read_records(tmp_path / "hf/hub", "reservations")
        assert records[0]["holders"] == 2

    @locks.guarded_operation
    def outer():
        locks._operation_stack.get().enter_context(locks.protect_caches(locations, repos))
        inner()
        assert worker.read_records(tmp_path / "hf/hub", "reservations")[0]["holders"] == 1

    outer()
    assert worker.read_records(tmp_path / "hf/hub", "reservations") == []
    assert len(worker.read_records(tmp_path / "hf/hub", "receipts")) == 1


def test_failed_or_cancelled_launch_retains_reservation(tmp_path):
    home = str(tmp_path / "hf")
    with pytest.raises(KeyboardInterrupt), locks.protect_caches([("localhost", home, {})], [{"repo_id": "org/model"}]):
        raise KeyboardInterrupt
    assert worker.read_records(tmp_path / "hf/hub", "reservations")


def test_launch_guard_dryrun_does_not_resolve_or_write(tmp_path):
    locks.reserve_launch(None, None, ["fake-host"], str(tmp_path / "hf"), "/unused", {}, dry_run=True)
    assert list(tmp_path.iterdir()) == []


def test_launch_and_build_entrypoints_are_guarded():
    from sparkrun.core.launcher import launch_inference
    from sparkrun.api._run import run
    from sparkrun.api._build import _prepare

    assert all(hasattr(function, "__wrapped__") for function in (launch_inference, run, _prepare))


def test_acquisition_failure_rolls_back_earlier_reservations_without_receipt(tmp_path, monkeypatch):
    from sparkrun.models import storage_remote

    real = storage_remote.query
    first, second = str(tmp_path / "a"), str(tmp_path / "b")

    def fail_second(host, home, action, **kwargs):
        if home == second and action == "reserve":
            raise WorkerError("second cache locked", "ModelInUse")
        return real(host, home, action, **kwargs)

    monkeypatch.setattr(storage_remote, "query", fail_second)
    with (
        pytest.raises(WorkerError),
        locks.protect_caches([("localhost", first, {}), ("localhost", second, {})], [{"repo_id": "org/model"}]),
    ):
        pytest.fail("preparation must not begin")
    assert worker.read_records(tmp_path / "a/hub", "reservations") == []
    assert worker.read_records(tmp_path / "a/hub", "receipts") == []
