"""Managed model claims: live instance identity, per-repo scope and races."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sparkrun.core.cluster_status import ClusterStatus, ContainerDetail, HostOccupancy, RunningWorkload
from sparkrun.core.model_usage import managed_usage, usage_blockers
from sparkrun.models import storage_worker as worker
from sparkrun.models.storage_remote import query, WorkerError
from sparkrun.core import model_storage_locks as locks
from test_model_storage import add_revision, COMMIT_A, COMMIT_B, selector


def live_job(instance="container-one", executor="docker"):
    return RunningWorkload(
        cluster_id="job-one",
        containers=(ContainerDetail("workload", "solo", "Up", "image", executor=executor, instance_id=instance),),
    )


def snapshot(*jobs, errors=None):
    return ClusterStatus(hosts=(HostOccupancy("localhost", workloads=jobs),), errors=errors or {})


def cache_record():
    return {
        "identity": "cache-one",
        "usage": {"reservations": []},
        "bindings": [
            {
                "job": "job-one",
                "identity": "cache-one",
                "instances": ["docker:container-one"],
                "recipe": "@test/recipe",
                "runtime": "vllm",
                "repos": [
                    {"repo_id": "org/model", "commits": [COMMIT_A]},
                    {"repo_id": "org/draft", "commits": None},
                ],
            }
        ],
    }


def selected(repo="org/model", commit=COMMIT_A):
    return [{"repo_id": repo, "commits": [commit]}]


def test_live_job_blocks_only_its_model_dependencies_and_pinned_revision():
    cache = cache_record()
    state = snapshot(live_job())
    for repo in ("org/model", "org/draft"):
        usage = managed_usage(state, "localhost", cache, selected(repo))
        assert usage["state"] == "in-use"
        assert "job-one" in usage_blockers(usage)[0] and "@test/recipe" in usage_blockers(usage)[0]
    for targets in (selected("org/unrelated"), selected(commit=COMMIT_B)):
        usage = managed_usage(state, "localhost", cache, targets)
        assert usage["state"] == "no-managed-consumers"
        assert not usage_blockers(usage)


@pytest.mark.parametrize("change", ["unbound", "restarted", "wrong-cache", "missing-instance", "malformed"])
def test_legacy_unverified_or_reused_job_identity_stays_unknown(change):
    cache = cache_record()
    job = live_job()
    if change == "unbound":
        cache["bindings"] = []
    elif change == "restarted":
        job = live_job("new-container")
    elif change == "wrong-cache":
        cache["bindings"][0]["identity"] = "another-cache"
    elif change == "missing-instance":
        job = live_job("")
    else:
        cache["bindings"][0]["repos"] = []
    usage = managed_usage(snapshot(job), "localhost", cache, selected("org/unrelated"))
    assert usage["state"] == "unknown"
    assert usage_blockers(usage)


def test_saved_binding_without_live_job_is_not_a_consumer():
    usage = managed_usage(snapshot(), "localhost", cache_record(), selected())
    assert usage["state"] == "no-managed-consumers"
    assert not usage_blockers(usage)


def test_whole_repository_use_does_not_depend_on_recognized_commits():
    for commits in ([], [COMMIT_B]):
        selection = [{"repo_id": "org/model", "commits": commits, "action": "remove-repository"}]
        usage = managed_usage(snapshot(live_job()), "localhost", cache_record(), selection)
        assert usage["state"] == "in-use"
        assert "job-one" in usage_blockers(usage)[0]


def test_incomplete_backend_observation_cannot_establish_nonuse():
    usage = managed_usage(snapshot(errors={"localhost": "local executor unavailable"}), "localhost", cache_record(), selected())
    assert usage["state"] == "unknown"
    assert "local executor unavailable" in usage_blockers(usage)[0]


def test_preparation_scope_and_legacy_uncertainty():
    cache = cache_record()
    cache["usage"]["reservations"] = [{"operation": "preparing-other", "repos": [{"repo_id": "org/other"}]}]
    assert managed_usage(snapshot(), "localhost", cache, selected())["state"] == "no-managed-consumers"
    assert managed_usage(snapshot(), "localhost", cache, selected("org/other"))["state"] == "preparing"
    usage = managed_usage(snapshot(), "localhost", cache, selected(), [{"operation": "legacy-download"}])
    assert usage["state"] == "unknown"
    covered = managed_usage(
        snapshot(), "localhost", cache, selected(), [{"operation": "model_download", "storage_operation": "preparing-other"}]
    )
    assert covered["state"] == "no-managed-consumers"


def test_worker_never_scans_arbitrary_processes_for_usage(tmp_path, monkeypatch):
    import subprocess

    monkeypatch.setattr(subprocess, "run", Mock(side_effect=AssertionError("no OS process scan")))
    usage = worker.consumers(tmp_path / "hub")
    assert usage["scope"] == "managed" and usage["uncertainty"] == []
    assert usage["state"] == "awaiting-managed-observation"


def test_worker_binding_requires_reservation_and_records_cached_pins(tmp_path):
    add_revision(tmp_path)
    repos = [{"repo_id": "org/model", "revision": COMMIT_A}]
    binding = dict(operation="op", job="job-one", recipe="recipe", runtime="vllm", repos=repos, instances=["docker:container-one"])
    with pytest.raises(WorkerError, match="reservation"):
        query("localhost", str(tmp_path), "bind", **binding)
    query("localhost", str(tmp_path), "reserve", operation="op", repos=repos)
    result = query("localhost", str(tmp_path), "bind", **binding)
    assert result["repos"][0]["commits"] == [COMMIT_A]
    assert result["repos"][0]["cached_revisions"] == [COMMIT_A]
    query("localhost", str(tmp_path), "release", operation="op")
    assert worker.read_records(tmp_path / "hub", "bindings") == [result]


def test_mutable_revision_binding_is_repository_scoped(tmp_path):
    add_revision(tmp_path)
    repos = [{"repo_id": "org/model", "revision": "main"}]
    query("localhost", str(tmp_path), "reserve", operation="op", repos=repos)
    result = query("localhost", str(tmp_path), "bind", operation="op", job="job", repos=repos, instances=["docker:id"])
    assert result["repos"][0]["commits"] is None
    assert result["repos"][0]["cached_revisions"] == [COMMIT_A]


def test_unrelated_preparation_does_not_block_selected_repo(tmp_path):
    add_revision(tmp_path)
    query("localhost", str(tmp_path), "reserve", operation="other", repos=[{"repo_id": "org/other"}])
    assert not worker.preview(tmp_path / "hub", [selector()])["blockers"]


def test_worker_refuses_remove_without_managed_check(tmp_path):
    add_revision(tmp_path)
    with pytest.raises(worker.StorageError, match="managed-workload check"):
        worker.remove_locked(tmp_path / "hub", {})


def test_bind_handoff_happens_before_preparation_guard_is_released(tmp_path, monkeypatch):
    from sparkrun.api import _status
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.recipe import Recipe
    from sparkrun.core.launcher import LaunchResult
    from sparkrun.runtimes.sglang import SglangRuntime

    add_revision(tmp_path)
    add_revision(tmp_path, repo="org/draft")
    recipe = Recipe.from_dict(
        {"name": "sample", "model": "org/model", "runtime": "sglang", "defaults": {"speculative_draft_model_path": "org/draft"}}
    )
    runtime = SglangRuntime()
    monkeypatch.setattr(_status, "status", lambda *args, **kwargs: snapshot(live_job()))
    result = SimpleNamespace(
        __class__=LaunchResult,
        rc=0,
        recipe=recipe,
        runtime=runtime,
        host_list=["localhost"],
        effective_cache_dir=str(tmp_path),
        storage_target={},
        cluster_id="job-one",
        config=SparkrunConfig(),
    )
    # Mock's spec preserves isinstance without constructing unrelated launch
    # fields. The reservations, binding writes and release are real workers.
    launch_result = Mock(spec=LaunchResult, **vars(result))

    @locks.guarded_operation
    def launch():
        locks.reserve_launch(recipe, runtime, ["localhost"], str(tmp_path), str(tmp_path), {})
        return launch_result

    launch()
    bindings = worker.read_records(tmp_path / "hub", "bindings")
    assert {r["repo_id"] for r in bindings[0]["repos"]} == {"org/model", "org/draft"}
    assert worker.read_records(tmp_path / "hub", "reservations") == []
    usage = managed_usage(
        snapshot(live_job()), "localhost", {"identity": worker.identity(tmp_path / "hub"), "bindings": bindings}, selected()
    )
    assert usage["state"] == "in-use"


def test_native_instance_identity_detects_pid_reuse():
    cache = cache_record()
    cache["bindings"][0]["instances"] = ["local:boot:100:200"]
    assert managed_usage(snapshot(live_job("boot:100:200", "local")), "localhost", cache, selected())["state"] == "in-use"
    assert managed_usage(snapshot(live_job("boot:100:300", "local")), "localhost", cache, selected())["state"] == "unknown"


def test_native_status_parser_retains_instance_identity():
    from sparkrun.orchestration.executors.local import _parse_local_pidfile_output

    jobs, _ = _parse_local_pidfile_output("sparkrun_abc123abc123abc1_def456abcdef_solo\t100\t\t\tboot:100:200\n")
    assert jobs[0].containers[0].instance_id == "boot:100:200"


def test_native_status_script_observes_real_process_start_identity(tmp_path):
    import os
    from pathlib import Path
    from sparkrun.orchestration.executors.local import LocalExecutor
    from sparkrun.orchestration.executors._base import ExecutorConfig

    name = "sparkrun_abc123abc123abc1_def456abcdef_solo"
    (tmp_path / (name + ".pid")).write_text(str(os.getpid()))
    executor = LocalExecutor(ExecutorConfig(pid_dir=str(tmp_path)))
    state = executor.query_status(["localhost"])
    assert not state.errors
    instance = state.hosts[0].workloads[0].containers[0].instance_id
    stat = Path("/proc/self/stat").read_text().rsplit(") ", 1)[1].split()
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    assert instance == "%s:%s:%s" % (boot, os.getpid(), stat[19])


@pytest.mark.parametrize("failure", ["rc", "binding"])
def test_unsuccessful_launch_or_handoff_retains_guards(tmp_path, monkeypatch, failure):
    from sparkrun.core.launcher import LaunchResult
    from sparkrun.core.recipe import Recipe
    from sparkrun.runtimes.sglang import SglangRuntime

    recipe = Recipe.from_dict({"name": "sample", "model": "org/model", "runtime": "sglang"})
    result = Mock(spec=LaunchResult, rc=1 if failure == "rc" else 0)
    monkeypatch.setattr(locks, "record_launch_usage", Mock(side_effect=WorkerError("binding unavailable")))

    @locks.guarded_operation
    def launch():
        locks.reserve_launch(recipe, SglangRuntime(), ["localhost"], str(tmp_path), str(tmp_path), {})
        return result

    if failure == "binding":
        with pytest.raises(WorkerError, match="binding unavailable"):
            launch()
    else:
        assert launch().rc == 1
    assert worker.read_records(tmp_path / "hub", "reservations")


def test_pending_marker_correlates_to_durable_storage_guard(tmp_path):
    from sparkrun.core.pending_ops import create_pending_op, list_pending_ops

    @locks.guarded_operation
    def prepare():
        create_pending_op("job", "model_download", cache_dir=str(tmp_path))
        assert list_pending_ops(str(tmp_path), read_only=True)[0]["storage_operation"] == locks._operation_id.get()

    prepare()


def test_detached_start_not_yet_observable_cannot_release_protection(monkeypatch):
    from sparkrun.api import _status
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.recipe import Recipe
    from sparkrun.runtimes.sglang import SglangRuntime

    recipe = Recipe.from_dict({"name": "sample", "model": "org/model", "runtime": "sglang"})
    result = SimpleNamespace(
        host_list=["localhost"],
        config=SparkrunConfig(),
        recipe=recipe,
        runtime=SglangRuntime(),
        cluster_id="job-one",
        storage_target={},
        storage_detached=True,
    )
    monkeypatch.setattr(_status, "status", lambda *args, **kwargs: snapshot())
    with pytest.raises(WorkerError, match="not observable"):
        locks.record_launch_usage(result)
    result.storage_detached = False
    locks.record_launch_usage(result)  # synchronous foreground execution ended
