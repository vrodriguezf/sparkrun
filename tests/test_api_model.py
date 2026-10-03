"""Console-free model API, target resolution, preflight and partial outcomes."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from sparkrun.api import model
from sparkrun.api.model import _ops
from sparkrun.models import storage_worker as worker
from sparkrun.models.storage_remote import WorkerError
from test_model_storage import add_revision, COMMIT_A, COMMIT_B, QUIET_USAGE
from test_model_storage_worker import FixtureSession


@pytest.fixture
def context(tmp_path):
    from sparkrun.api import default_sctx

    sctx = default_sctx()
    sctx.config._data["hf_cache_dir"] = str(tmp_path / "hf")
    return sctx


@pytest.fixture
def isolated_workers(monkeypatch):
    monkeypatch.setattr(worker, "consumers", lambda hub: dict(QUIET_USAGE))

    def query(host, hf_home, action, **kwargs):
        kwargs.pop("ssh_kwargs", None)
        try:
            return worker.dispatch({"schema_version": 1, "hf_home": hf_home, "action": action, **kwargs})
        except worker.StorageError as exc:
            raise WorkerError(str(exc), exc.code) from exc

    monkeypatch.setattr(_ops, "query", query)
    monkeypatch.setattr(_ops, "WorkerSession", FixtureSession)
    from sparkrun.api import _status
    from sparkrun.core.cluster_status import ClusterStatus, HostOccupancy

    monkeypatch.setattr(_status, "status", lambda hosts, **kwargs: ClusterStatus(hosts=tuple(HostOccupancy(h) for h in hosts)))


def test_local_inventory_is_offline_readonly_and_console_free(context, isolated_workers, capsys):
    home = context.config.hf_cache_dir
    add_revision(home)
    before = list(home.rglob("*"))
    report = model.list_cached(model.StorageOptions(local=True), sctx=context)
    assert report.complete
    assert report.caches[0]["models"][0]["repo_id"] == "org/model"
    assert list(home.rglob("*")) == before
    assert capsys.readouterr() == ("", "")
    json.dumps(report.to_dict())


def test_recipe_pin_and_dependency_opt_in(context, isolated_workers):
    from sparkrun.core.recipe import Recipe

    home = context.config.hf_cache_dir
    add_revision(home, repo="org/primary")
    add_revision(home, repo="org/draft", commit=COMMIT_B)
    recipe = Recipe.from_dict(
        {
            "name": "sample",
            "model": "org/primary",
            "model_revision": COMMIT_A,
            "runtime": "sglang",
            "defaults": {"speculative_draft_model_path": "org/draft", "speculative_draft_model_revision": COMMIT_B},
        }
    )
    options = model.StorageOptions(local=True)
    primary = model.plan_remove(model.ModelSelector(recipe=recipe), options, sctx=context)
    assert len(primary.targets[0].snapshot["selected"]) == 1
    assert primary.targets[0].snapshot["selected"][0]["commits"] == [COMMIT_A]
    all_inputs = model.plan_remove(model.ModelSelector(recipe=recipe, include_dependencies=True), options, sctx=context)
    assert [s["commits"] for s in all_inputs.targets[0].snapshot["selected"]] == [[COMMIT_A], [COMMIT_B]]


@pytest.mark.parametrize(
    "selector",
    [
        model.ModelSelector(),
        model.ModelSelector(model="org/model", recipe="sample"),
        model.ModelSelector(recipe="sample", revision="main"),
        model.ModelSelector(model="org/model:Q4_K_M"),
        model.ModelSelector(model="/mnt/weights"),
    ],
)
def test_bad_or_broader_removal_selection_refused(context, selector):
    with pytest.raises(model.InvalidModelSelection):
        model.plan_remove(selector, model.StorageOptions(local=True), sctx=context)


def test_remote_head_cache_shared_by_worker_resolution(context, monkeypatch):
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core import launcher

    probe = Mock(return_value="/head/custom-hf")
    monkeypatch.setattr(launcher, "resolve_effective_cache_dir", probe)
    cluster = ClusterDefinition(name="two", hosts=["node-a", "node-b"], user="model-user")
    locations, _, _ = _ops._locations(model.StorageOptions(cluster=cluster), context)
    assert [loc.hf_home for loc in locations] == ["/head/custom-hf", "/head/custom-hf"]
    assert all(loc.ssh_kwargs["ssh_user"] == "model-user" for loc in locations)
    assert context.config.ssh_user != "model-user"


def test_no_default_falls_back_local_and_deduplicates(context, isolated_workers):
    report = model.list_cached(model.StorageOptions(include_local=True), sctx=context)
    assert len(report.caches) == 1
    assert "default selection" in report.caches[0]["location"]["source"]


def test_readonly_preview_no_cache_creation(context, isolated_workers):
    home = context.config.hf_cache_dir
    plan = model.plan_remove(model.ModelSelector(model="org/model"), model.StorageOptions(local=True), sctx=context)
    assert not plan.blockers
    assert not home.exists()
    result = model.remove(plan, sctx=context)
    assert result.complete and result.results[0]["state"] == "already-absent"
    assert not home.exists()


def test_remove_revalidates_and_repeated_remove_is_noop(context, isolated_workers):
    home = context.config.hf_cache_dir
    add_revision(home)
    options = model.StorageOptions(local=True)
    selector = model.ModelSelector(model="org/model")
    plan = model.plan_remove(selector, options, sctx=context)
    add_revision(home, commit=COMMIT_B, blob="new")
    with pytest.raises(model.RemovalPlanChanged):
        model.remove(plan, sctx=context)
    result = model.remove(model.plan_remove(selector, options, sctx=context), sctx=context)
    assert result.complete and result.results[0]["state"] == "removed"
    result = model.remove(model.plan_remove(selector, options, sctx=context), sctx=context)
    assert result.results[0]["state"] == "already-absent"


def test_failed_preflight_prevents_all_deletion(context, isolated_workers, tmp_path):
    first = context.config.hf_cache_dir
    second = tmp_path / "second"
    add_revision(first)
    add_revision(second)
    options = model.StorageOptions(local=True, include_local=True, local_cache_dir=str(second), cache_dir=str(first))
    plan = model.plan_remove(model.ModelSelector(model="org/model"), options, sctx=context)
    assert len(plan.targets) == 2
    add_revision(second, commit=COMMIT_B, blob="new")
    with pytest.raises(model.RemovalPlanChanged):
        model.remove(plan, sctx=context)
    assert worker.scan(first / "hub")["models"]


def test_valid_and_damaged_caches_share_one_removal_plan(context, isolated_workers, tmp_path):
    first = context.config.hf_cache_dir
    second = tmp_path / "second"
    damaged = add_revision(first)
    healthy = add_revision(second)
    (damaged / "refs/main").write_text(COMMIT_B)
    options = model.StorageOptions(local=True, include_local=True, local_cache_dir=str(second), cache_dir=str(first))
    plan = model.plan_remove(model.ModelSelector(model="org/model"), options, sctx=context)
    assert not plan.blockers and len(plan.targets) == 2
    rows = [t.snapshot["selected"][0] for t in plan.targets]
    assert {r["cache_state"] for r in rows} == {"cached", "corrupt-or-incomplete"}
    assert all(r["action"] == "remove-repository" and r["allocated_bytes"] > 0 for r in rows)
    result = model.remove(plan, sctx=context)
    assert result.complete
    assert not damaged.exists() and not healthy.exists()


def test_blocked_target_remains_visible_and_prevents_other_cache_deletion(context, isolated_workers, tmp_path):
    first = context.config.hf_cache_dir
    second = tmp_path / "second"
    damaged = add_revision(first)
    healthy = add_revision(second)
    (damaged / "refs/main").write_text(COMMIT_B)
    options = model.StorageOptions(local=True, include_local=True, local_cache_dir=str(second), cache_dir=str(first))
    plan = model.plan_remove(model.ModelSelector(model="org/model", revision=COMMIT_A), options, sctx=context)
    assert plan.blockers and len(plan.targets) == 2
    assert {t.snapshot["selected"][0]["action"] for t in plan.targets} == {"blocked", "remove-revisions"}
    with pytest.raises(model.ModelInUse):
        model.remove(plan, sctx=context)
    assert damaged.exists() and healthy.exists()


def test_unreachable_target_remains_in_plan(context, isolated_workers, monkeypatch):
    def unavailable(*args, **kwargs):
        raise WorkerError("SSH unavailable")

    monkeypatch.setattr(_ops, "_query", unavailable)
    plan = model.plan_remove(model.ModelSelector(model="org/model"), model.StorageOptions(local=True), sctx=context)
    assert plan.blockers and len(plan.targets) == 1
    row = plan.to_dict()["targets"][0]["selected"][0]
    assert row["action"] == "blocked" and row["allocated_bytes"] is None
    assert "SSH unavailable" in row["error"]


def test_damaged_cache_blocks_live_job_even_with_no_recognized_commits(context, isolated_workers, monkeypatch):
    from sparkrun.api import _status
    from test_model_storage_usage import live_job, snapshot, cache_record

    home = context.config.hf_cache_dir
    repo = add_revision(home)
    (repo / "refs/main").write_text(COMMIT_B)
    binding = cache_record()["bindings"][0]
    binding["identity"] = worker.identity(home / "hub")
    worker.write_record(home / "hub", "bindings", "job-one", binding)
    monkeypatch.setattr(_status, "status", lambda *args, **kwargs: snapshot(live_job()))
    plan = model.plan_remove(model.ModelSelector(model="org/model"), model.StorageOptions(local=True), sctx=context)
    assert any("job-one" in b for b in plan.blockers)
    assert plan.targets[0].snapshot["usage"]["state"] == "in-use"
    assert repo.exists()


def test_partial_reply_loss_is_not_success(context, isolated_workers, tmp_path, monkeypatch):
    first = context.config.hf_cache_dir
    second = tmp_path / "second"
    add_revision(first)
    add_revision(second)
    options = model.StorageOptions(local=True, include_local=True, local_cache_dir=str(second), cache_dir=str(first))
    plan = model.plan_remove(model.ModelSelector(model="org/model"), options, sctx=context)
    actual = FixtureSession.request
    count = 0

    def failing(self, action, **kwargs):
        nonlocal count
        if action == "remove":
            count += 1
            if count == 2:
                raise WorkerError("Connection lost")
        return actual(self, action, **kwargs)

    monkeypatch.setattr(FixtureSession, "request", failing)
    result = model.remove(plan, sctx=context)
    assert not result.complete
    assert [r["state"] for r in result.results] == ["removed", "indeterminate"]


def test_managed_job_allows_removing_other_model(context, isolated_workers, monkeypatch):
    from sparkrun.api import _status
    from test_model_storage_usage import live_job, snapshot

    home = context.config.hf_cache_dir
    active = add_revision(home)
    other = add_revision(home, repo="org/other")
    worker.write_record(
        home / "hub",
        "bindings",
        "job-one",
        {
            "job": "job-one",
            "identity": worker.identity(home / "hub"),
            "instances": ["docker:container-one"],
            "recipe": "sample",
            "repos": [{"repo_id": "org/model", "commits": [COMMIT_A]}],
        },
    )
    monkeypatch.setattr(_status, "status", lambda *args, **kwargs: snapshot(live_job()))
    options = model.StorageOptions(local=True)
    inventory = model.list_cached(options, sctx=context)
    assert {m["repo_id"]: m["usage"]["state"] for m in inventory.caches[0]["models"]} == {
        "org/model": "in-use",
        "org/other": "no-managed-consumers",
    }
    protected = model.plan_remove(model.ModelSelector(model="org/model"), options, sctx=context)
    assert any("job-one" in b for b in protected.blockers)
    plan = model.plan_remove(model.ModelSelector(model="org/other"), options, sctx=context)
    assert not plan.blockers
    assert model.remove(plan, sctx=context).complete
    assert active.exists() and not other.exists()


@pytest.mark.parametrize("damaged", [False, True])
def test_job_starting_after_preview_is_rechecked_under_lock(context, isolated_workers, monkeypatch, damaged):
    from sparkrun.api import _status
    from sparkrun.models.storage_remote import query
    from test_model_storage_usage import live_job, snapshot

    home = context.config.hf_cache_dir
    repo = add_revision(home)
    if damaged:
        (repo / "refs/main").write_text(COMMIT_B)
    plan = model.plan_remove(model.ModelSelector(model="org/model"), model.StorageOptions(local=True), sctx=context)

    def observe(*args, **kwargs):
        assert kwargs["read_only"] is True
        # Every managed launcher must acquire this same target-side guard.
        with pytest.raises(WorkerError, match="holds this cache"):
            query("localhost", str(home), "reserve", operation="new-launch", repos=[{"repo_id": "org/model"}])
        return snapshot(live_job())  # legacy/unbound job that appeared meanwhile

    monkeypatch.setattr(_status, "status", observe)
    with pytest.raises(model.ModelInUse, match="no verified model binding"):
        model.remove(plan, sctx=context)
    assert repo.exists()


def test_later_host_usage_failure_prevents_any_deletion(context, isolated_workers, monkeypatch, tmp_path):
    from sparkrun.api import _status
    from sparkrun.core.cluster_status import ClusterStatus

    home = context.config.hf_cache_dir
    second = tmp_path / "second"
    first_repo, second_repo = add_revision(home), add_revision(second)
    plan = model.plan_remove(
        model.ModelSelector(model="org/model"),
        model.StorageOptions(local=True, include_local=True, cache_dir=str(home), local_cache_dir=str(second)),
        sctx=context,
    )
    from test_model_storage_usage import snapshot

    checks = iter([snapshot(), ClusterStatus(errors={"localhost": "status unavailable"})])
    monkeypatch.setattr(_status, "status", lambda *args, **kwargs: next(checks))
    with pytest.raises(model.ModelInUse, match="status unavailable"):
        model.remove(plan, sctx=context)
    assert first_repo.exists() and second_repo.exists()


def test_recorded_native_namespace_is_probed_even_if_default_is_idle(context, isolated_workers, monkeypatch):
    from sparkrun.api import _status
    from test_model_storage_usage import snapshot, live_job

    home = context.config.hf_cache_dir
    add_revision(home)
    worker.write_record(
        home / "hub",
        "bindings",
        "job-one",
        {
            "job": "job-one",
            "identity": worker.identity(home / "hub"),
            "instances": ["local:boot:100:200"],
            "recipe": "native",
            "target": {"executor": "local", "pid_dir": "/custom/pids"},
            "repos": [{"repo_id": "org/model", "commits": None}],
        },
    )
    calls = []

    def status(hosts, *, cluster, **kwargs):
        calls.append(cluster.executor_config)
        return (
            snapshot(live_job("boot:100:200", "local")) if (cluster.executor_config or {}).get("pid_dir") == "/custom/pids" else snapshot()
        )

    monkeypatch.setattr(_status, "status", status)
    plan = model.plan_remove(model.ModelSelector(model="org/model"), model.StorageOptions(local=True), sctx=context)
    assert {"pid_dir": "/custom/pids"} in calls
    assert any("job-one" in b for b in plan.blockers)


def test_selected_cluster_namespace_is_preserved(context, isolated_workers, monkeypatch):
    from sparkrun.api import _status
    from sparkrun.core.cluster_manager import ClusterDefinition
    from test_model_storage_usage import snapshot

    cluster = ClusterDefinition(name="native", hosts=["localhost"], executor="local", executor_config={"pid_dir": "/native/pids"})
    probe = Mock(return_value=snapshot())
    monkeypatch.setattr(_status, "status", probe)
    model.list_cached(model.StorageOptions(cluster=cluster), sctx=context)
    assert probe.call_args.kwargs["cluster"].executor_config == {"pid_dir": "/native/pids"}


def test_correlated_preparation_of_other_repo_does_not_block_removal(context, isolated_workers):
    from sparkrun.core import model_storage_locks as locks
    from sparkrun.core.pending_ops import create_pending_op

    home = context.config.hf_cache_dir
    add_revision(home)

    @locks.guarded_operation
    def prepare_other():
        with locks.protect_caches([("localhost", str(home), {})], [{"repo_id": "org/other"}]):
            create_pending_op("job-other", "model_download", model="org/other", cache_dir=str(context.config.cache_dir))
            plan = model.plan_remove(model.ModelSelector(model="org/model"), model.StorageOptions(local=True), sctx=context)
            assert not plan.blockers

    prepare_other()


def test_pull_plan_pins_dependencies_and_does_not_prepare_runtime(context, monkeypatch):
    from sparkrun.core.recipe import Recipe
    from sparkrun.models import revision
    from sparkrun.runtimes.sglang import SglangRuntime

    recipe = Recipe.from_dict(
        {
            "name": "sample",
            "model": "org/primary",
            "model_revision": COMMIT_A,
            "runtime": "sglang",
            "defaults": {"speculative_draft_model_path": "org/draft", "speculative_draft_model_revision": COMMIT_B},
        }
    )
    monkeypatch.setattr(SglangRuntime, "prepare", Mock(side_effect=AssertionError("runtime preparation is forbidden")))
    resolve = Mock(side_effect=lambda repo, rev, **kwargs: rev)
    monkeypatch.setattr(revision, "resolve_hub_commit", resolve)
    plan = model.plan_pull(model.ModelSelector(recipe=recipe), model.StorageOptions(local=True), sctx=context)
    assert [item["revision"] for item in plan.inputs] == [COMMIT_A, COMMIT_B]
    assert all(item["target"] == [0] for item in plan.inputs)


def test_pull_failure_exit_code_not_reported_success(context, monkeypatch):
    from sparkrun.models import download

    download_mock = Mock(return_value=1)
    monkeypatch.setattr(download, "download_model", download_mock)
    monkeypatch.setattr(_ops, "setup_helper", lambda *args, **kwargs: None)
    plan = model.plan_pull(model.ModelSelector(model="org/model", revision=COMMIT_A), model.StorageOptions(local=True), sctx=context)
    result = model.pull(plan, sctx=context)
    assert not result.complete
    assert result.results[0]["state"] == "failed"
    assert download_mock.called


def test_local_pull_records_actual_revision_without_images_or_scheduling(context, monkeypatch):
    from sparkrun.models import download
    from sparkrun.api import _run
    from sparkrun.core import image_preparation

    def fake_download(repo, *, cache_dir, revision, token):
        add_revision(Path(cache_dir), repo=repo, commit=revision)
        return 0

    monkeypatch.setattr(download, "download_model", fake_download)
    monkeypatch.setattr(_run, "plan", Mock(side_effect=AssertionError("must not schedule")))
    monkeypatch.setattr(image_preparation, "prepare_images", Mock(side_effect=AssertionError("must not prepare images")))
    plan = model.plan_pull(model.ModelSelector(model="org/model", revision=COMMIT_A), model.StorageOptions(local=True), sctx=context)
    result = model.pull(plan, sctx=context)
    assert result.complete and result.results[0]["state"] == "pulled"
    receipts = worker.read_records(context.config.hf_cache_dir / "hub", "receipts")
    assert receipts[0]["cached_revisions"] == {"org/model": [COMMIT_A]}
    assert worker.read_records(context.config.hf_cache_dir / "hub", "reservations") == []


def test_api_has_no_click_dependency():
    import ast

    for path in Path(_ops.__file__).parent.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("click")
            elif isinstance(node, ast.Import):
                assert all(not n.name.startswith("click") for n in node.names)


def test_recipe_file_resolution_is_offline_and_preserves_catalog_trust(context, tmp_path, monkeypatch):
    from sparkrun.api import SparkrunError
    from sparkrun.core._recipe_source import catalog_root

    imported = catalog_root(context.config) / "imports" / "untrusted.yaml"
    imported.parent.mkdir(parents=True)
    imported.write_text("name: untrusted\nmodel: org/model\nruntime: vllm\ncluster_config:\n  remote_cache_dir: /unexpected/hf\n")
    monkeypatch.setattr(_ops, "query", Mock(side_effect=AssertionError("must reject before probing cache")))
    with pytest.raises(SparkrunError, match="trust|Trust|Untrusted|untrusted"):
        model.plan_remove(model.ModelSelector(recipe=str(imported)), model.StorageOptions(local=True), sctx=context)


def test_distribution_disabled_never_contacts_hub(context, monkeypatch):
    from sparkrun.core.recipe import Recipe
    from sparkrun.models import revision

    recipe = Recipe.from_dict({"name": "disabled", "model": "org/model", "runtime": "vllm", "distribution": {"models": {"enabled": False}}})
    recipe.distribution_config.models.enabled = False
    monkeypatch.setattr(revision, "resolve_hub_commit", Mock(side_effect=AssertionError("disabled download must not query Hub")))
    plan = model.plan_pull(model.ModelSelector(recipe=recipe), model.StorageOptions(local=True), sctx=context)
    assert not plan.enabled
    assert model.pull(plan, sctx=context).results[0]["state"] == "disabled"


def test_external_path_pull_does_not_create_cache_or_helper(context, monkeypatch):
    monkeypatch.setattr(_ops, "setup_helper", Mock(side_effect=AssertionError("external paths need no helper")))
    plan = model.plan_pull(model.ModelSelector(model="/opt/preplaced"), model.StorageOptions(local=True), sctx=context)
    assert model.pull(plan, sctx=context).results[0]["state"] == "externally-managed"
    assert not context.config.hf_cache_dir.exists()


def test_invalid_target_indices_validated_before_hub(context, monkeypatch):
    from sparkrun.core.recipe import Recipe, DistributionModelEntry
    from sparkrun.models import revision

    recipe = Recipe.from_dict({"name": "bad-target", "model": "org/model", "runtime": "vllm"})
    recipe.distribution_config.models.entries = [DistributionModelEntry(name="org/model", target=[1])]
    monkeypatch.setattr(revision, "resolve_hub_commit", Mock(side_effect=AssertionError("must not query Hub")))
    with pytest.raises(model.InvalidModelSelection, match="index"):
        model.plan_pull(model.ModelSelector(recipe=recipe), model.StorageOptions(local=True), sctx=context)


def test_broken_default_does_not_fall_back_to_local_or_clear_marker(context):
    from sparkrun.api import SparkrunError

    marker = context.config.config_path.parent / "clusters/.default"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("deleted-cluster")
    with pytest.raises(SparkrunError):
        model.list_cached(sctx=context)
    assert marker.read_text() == "deleted-cluster"


def test_local_recipe_uses_local_override_not_remote_home(context):
    from sparkrun.core.recipe import Recipe

    recipe = Recipe.from_dict(
        {
            "name": "paths",
            "model": "org/model",
            "runtime": "vllm",
            "cluster_config": {"local_cache_dir": "/control/hf", "remote_cache_dir": "/nodes/hf"},
        }
    )
    locations, _, _ = _ops._locations(model.StorageOptions(local=True), context, recipe)
    assert locations[0].hf_home == "/control/hf"
