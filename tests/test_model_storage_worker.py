"""Run real worker sessions and advisory exclusion in separate processes."""

import json
import subprocess
import sys

import pytest

from sparkrun.models.storage_remote import WorkerSession, WorkerError, query
from sparkrun.models import storage_worker as worker
from test_model_storage import add_revision, selector, QUIET_USAGE


class FixtureSession(WorkerSession):
    """Keep filesystem/locking real; isolate OS consumer observation."""

    def __enter__(self):
        from queue import Queue
        import threading

        code = ("from sparkrun.models import storage_worker as w;w.consumers=lambda hub: %r;w.main()") % QUIET_USAGE
        self.process = subprocess.Popen(
            [sys.executable, "-u", "-c", code], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        self.responses = Queue()

        def reader():
            for line in self.process.stdout:
                self.responses.put(line)
            self.responses.put(None)

        threading.Thread(target=reader, daemon=True).start()
        return self


def test_real_worker_protocol_and_receipt(tmp_path):
    home = str(tmp_path / "hf")
    assert query("localhost", home, "reserve", operation="test", repos=[{"repo_id": "org/model"}])["state"] == "reserved"
    assert query("localhost", home, "release", operation="test", receipt={"operation": "test"})["state"] == "released"
    receipts = worker.read_records(tmp_path / "hf/hub", "receipts")
    assert receipts == [{"operation": "test", "cached_revisions": {}}]


def test_two_processes_exclude_delete_and_reserve(tmp_path, monkeypatch):
    add_revision(tmp_path)
    monkeypatch.setattr(worker, "consumers", lambda hub: dict(QUIET_USAGE))
    expected = worker.preview(tmp_path / "hub", [selector()])
    with FixtureSession("localhost", str(tmp_path)) as first:
        first.request("lock", selectors=[selector()], expected=expected)
        with pytest.raises(WorkerError, match="holds this cache"):
            query("localhost", str(tmp_path), "reserve", operation="second", repos=[])
        with FixtureSession("localhost", str(tmp_path)) as second:
            with pytest.raises(WorkerError, match="holds this cache"):
                second.request("lock", selectors=[selector()], expected=expected)
        result = first.request(
            "remove", selectors=[selector()], expected=expected, operation="delete-test", usage_scope="managed", managed_checked=True
        )
        assert result["state"] == "removed"
    assert worker.scan(tmp_path / "hub")["models"] == []


def test_eof_releases_lock_and_durable_reservation_survives_crash(tmp_path, monkeypatch):
    add_revision(tmp_path)
    monkeypatch.setattr(worker, "consumers", lambda hub: dict(QUIET_USAGE))
    expected = worker.preview(tmp_path / "hub", [selector()])
    with FixtureSession("localhost", str(tmp_path)) as session:
        session.request("lock", selectors=[selector()], expected=expected)
    query("localhost", str(tmp_path), "reserve", operation="crashed", repos=[])
    assert worker.read_records(tmp_path / "hub", "reservations")[0]["operation"] == "crashed"
    # Acquiring again demonstrates the inode lock went away, not the receipt.
    fd = worker.guard(tmp_path / "hub")
    import os

    os.close(fd)
    with FixtureSession("localhost", str(tmp_path)) as session:
        with pytest.raises(WorkerError, match="preparing"):
            session.request("lock", selectors=[selector()], expected=expected)


def test_worker_rejects_wrong_protocol_and_unlocked_deletion(tmp_path):
    with FixtureSession("localhost", str(tmp_path)) as session:
        with pytest.raises(WorkerError, match="locked session"):
            session.request("remove", selectors=[selector()])
        session.process.stdin.write((json.dumps({"schema_version": 999, "action": "scan", "hf_home": str(tmp_path)}) + "\n").encode())
        session.process.stdin.flush()
        assert json.loads(session.responses.get(timeout=5))["error"]["message"] == "Incompatible model storage protocol"


def test_disconnected_worker_cleanup_preserves_original_error():
    from unittest.mock import Mock

    session = WorkerSession("localhost", "/unused/hf")
    session.process = Mock()
    session.process.stdin.close.side_effect = BrokenPipeError("worker disconnected")
    session.__exit__(None, None, None)
    session.process.wait.assert_called_once_with(timeout=5)
    session.process.stdout.close.assert_called_once()


def test_local_setup_preserves_capability_error(monkeypatch):
    from sparkrun.models.storage_remote import setup_helper

    def unavailable():
        raise worker.StorageError("incompatible backend")

    monkeypatch.setattr(worker, "backend", unavailable)
    with pytest.raises(WorkerError) as error:
        setup_helper("localhost")
    assert error.value.code == "StorageCapabilityMissing"
