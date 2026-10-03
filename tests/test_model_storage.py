"""Filesystem safety tests: every byte under test lives in a temporary cache."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from sparkrun.models import storage_worker as worker
from sparkrun.models.download import model_cache_path


COMMIT_A = "a" * 40
COMMIT_B = "b" * 40
QUIET_USAGE = {"state": "not-observed-in-use", "consumers": [], "uncertainty": [], "reservations": [], "coverage": "test fixture"}


def add_revision(home, repo="org/model", commit=COMMIT_A, blob="weights", content=b"weights", ref="main"):
    path = Path(model_cache_path(repo, str(home)))
    (path / "blobs").mkdir(parents=True, exist_ok=True)
    payload = path / "blobs" / blob
    if not payload.exists():
        payload.write_bytes(content)
    snapshot = path / "snapshots" / commit
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "model.safetensors").symlink_to("../../blobs/" + blob)
    if ref:
        (path / "refs").mkdir(exist_ok=True)
        (path / "refs" / ref).write_text(commit)
    return path


def selector(repo="org/model", revision=None):
    return {"repo_id": repo, "revision": revision, "cache_name": Path(model_cache_path(repo, "/fixture")).name}


@pytest.fixture
def cache(tmp_path, monkeypatch):
    home = tmp_path / "hf"
    home.mkdir()
    monkeypatch.setattr(worker, "consumers", lambda hub: dict(QUIET_USAGE))
    return home


def delete(home, selectors, expected=None):
    hub = home / "hub"
    expected = expected or worker.preview(hub, selectors)
    return worker.remove_locked(
        hub,
        {"selectors": selectors, "expected": expected, "operation": "fixture-removal", "usage_scope": "managed", "managed_checked": True},
    )


def test_missing_cache_inventory_and_preview_are_read_only(cache):
    assert worker.scan(cache / "hub")["models"] == []
    preview = worker.preview(cache / "hub", [selector()])
    assert preview["selected"][0]["state"] == "already-absent"
    assert not (cache / "hub").exists()


def test_inventory_does_not_need_recipe_or_job_metadata(cache):
    add_revision(cache)
    add_revision(cache, commit=COMMIT_B, blob="new", content=b"other weights", ref="stable")
    report = worker.scan(cache / "hub")
    assert report["models"][0]["repo_id"] == "org/model"
    assert len(report["models"][0]["revisions"]) == 2
    assert report["receipts"] == []
    assert report["filesystem"]["free_bytes"] > 0


def test_revision_delete_keeps_shared_blob_and_surviving_ref(cache):
    repo = add_revision(cache)
    add_revision(cache, commit=COMMIT_B, ref="stable")
    result = delete(cache, [selector(revision="main")])
    assert result["state"] == "removed"
    assert not (repo / "snapshots" / COMMIT_A).exists()
    assert (repo / "snapshots" / COMMIT_B / "model.safetensors").read_bytes() == b"weights"
    assert not (repo / "refs/main").exists()
    assert (repo / "refs/stable").exists()


def test_same_commit_other_repo_never_broadens_selection(cache):
    first = add_revision(cache)
    second = add_revision(cache, repo="other/model")
    delete(cache, [selector()])
    assert not first.exists()
    assert (second / "snapshots" / COMMIT_A / "model.safetensors").is_file()


def test_multiple_selectors_for_one_repo_are_one_reference_aware_strategy(cache):
    repo = add_revision(cache)
    add_revision(cache, commit=COMMIT_B, ref="stable")
    selectors = [selector(revision="main"), selector(revision="stable")]
    selected, strategies = worker.selection(cache / "hub", selectors)
    assert len(selected) == len(strategies) == 1
    assert selected[0]["commits"] == [COMMIT_A, COMMIT_B]
    delete(cache, selectors)
    assert repo.exists()  # Revision selectors do not delete unrelated repo metadata.
    assert not list((repo / "snapshots").iterdir())


def test_absent_is_noop_but_unknown_revision_is_error(cache):
    add_revision(cache)
    assert worker.preview(cache / "hub", [selector("absent/model")])["selected"][0]["state"] == "already-absent"
    report = worker.preview(cache / "hub", [selector(revision="missing")])
    assert "not cached" in report["blockers"][0]
    assert report["selected"][0]["action"] == "blocked"


def test_corrupt_repo_is_visible_measured_and_removable(cache):
    path = cache / "hub/models--broken--model/blobs"
    path.mkdir(parents=True)
    (path / "payload.incomplete").write_bytes(b"unfinished")
    report = worker.scan(cache / "hub")
    assert report["models"][0]["state"] == "corrupt-or-incomplete"
    assert report["models"][0]["attributed_bytes"] is None
    preview = worker.preview(cache / "hub", [selector("broken/model")])
    row = preview["selected"][0]
    assert not preview["blockers"]
    assert row["action"] == "remove-repository" and not row["commits"]
    assert row["contents"]["partial_files"] == ["blobs/payload.incomplete"]
    assert row["allocated_bytes"] == row["reclaimable_bytes"] > 0
    assert delete(cache, [selector("broken/model")])["state"] == "removed"
    assert not path.parent.exists()


def test_shared_blob_store_refused_without_deleting_other_model(cache):
    repo = add_revision(cache)
    store = cache / "hub/blobs/aa"
    store.mkdir(parents=True)
    (store / "shared").write_bytes(b"shared data")
    (repo / "blobs/weights").unlink()
    (repo / "blobs/weights").symlink_to("../../blobs/aa/shared")
    assert any("Shared/external" in b for b in worker.preview(cache / "hub", [selector()])["blockers"])
    assert (store / "shared").read_bytes() == b"shared data"


def test_untracked_escaping_link_blocks_recursive_repository_removal(cache, tmp_path):
    repo = add_revision(cache)
    outside = tmp_path / "private"
    outside.write_bytes(b"keep me")
    (repo / "untracked").symlink_to(outside)
    assert any("Shared/external" in b for b in worker.preview(cache / "hub", [selector()])["blockers"])
    assert outside.read_bytes() == b"keep me"


def test_stale_plan_detects_ref_changes_and_repository_replacement(cache):
    repo = add_revision(cache)
    expected = worker.preview(cache / "hub", [selector()])
    add_revision(cache, commit=COMMIT_B, blob="new")
    with pytest.raises(worker.StorageError, match="changed after preview"):
        delete(cache, [selector()], expected)
    assert repo.exists()


def test_library_silent_delete_failure_is_not_reported_success(cache, monkeypatch):
    add_revision(cache)
    from huggingface_hub import DeleteCacheStrategy

    monkeypatch.setattr(DeleteCacheStrategy, "execute", lambda self: None)
    with pytest.raises(worker.StorageError, match="verification failed"):
        delete(cache, [selector(revision="main")])
    journal = list((cache / "hub/.locks/sparkrun/journal").glob("*.json"))
    assert json.loads(journal[0].read_text())["state"] == "started"


def test_preparing_selected_repo_blocks_removal(cache):
    add_revision(cache)
    worker.write_record(cache / "hub", "reservations", "one", {"operation": "one", "repos": [{"repo_id": "org/model"}]})
    assert worker.preview(cache / "hub", [selector()])["blockers"]
    with pytest.raises(worker.StorageError):
        delete(cache, [selector()])


def test_dangling_ref_keeps_physical_snapshot_visible_and_whole_repo_removable(cache):
    repo = add_revision(cache)
    (repo / "refs/main").write_text(COMMIT_B)
    row = worker.preview(cache / "hub", [selector()])["selected"][0]
    assert row["cache_state"] == "corrupt-or-incomplete"
    assert row["contents"]["snapshots"] == [{"commit": COMMIT_A, "files": ["model.safetensors"]}]
    assert COMMIT_B in row["issues"][0]
    for revision in ("main", COMMIT_A):
        preview = worker.preview(cache / "hub", [selector(revision=revision)])
        assert preview["blockers"] and "without --revision" in preview["blockers"][0]
    assert delete(cache, [selector()])["state"] == "removed"
    assert not repo.exists()


def test_whole_repo_includes_orphans_but_keeps_other_models_and_records(cache):
    repo = add_revision(cache)
    other = add_revision(cache, repo="other/model")
    (repo / "blobs/orphan").write_bytes(b"x" * 8192)
    (repo / "blobs/interrupted.incomplete").write_bytes(b"partial")
    worker.write_record(cache / "hub", "receipts", "historic", {"operation": "historic"})
    preview = worker.preview(cache / "hub", [selector()])
    assert preview["selected"][0]["allocated_bytes"] >= 8192
    delete(cache, [selector()], preview)
    assert not repo.exists() and other.exists()
    assert worker.read_records(cache / "hub", "receipts") == [{"operation": "historic"}]


def test_last_revision_does_not_remove_orphans_or_extra_metadata(cache):
    repo = add_revision(cache)
    (repo / "blobs/orphan").write_bytes(b"keep orphan")
    (repo / "notes").write_text("keep metadata")
    preview = worker.preview(cache / "hub", [selector(revision="main")])
    assert preview["selected"][0]["action"] == "remove-revisions"
    delete(cache, [selector(revision="main")], preview)
    assert not (repo / "snapshots" / COMMIT_A).exists()
    assert not (repo / "blobs/weights").exists()
    assert (repo / "notes").read_text() == "keep metadata"
    assert (repo / "blobs/orphan").read_bytes() == b"keep orphan"


def test_corrupt_cache_without_commits_still_blocks_preparation(cache):
    repo = add_revision(cache)
    (repo / "refs/main").write_text(COMMIT_B)
    worker.write_record(cache / "hub", "reservations", "active", {"operation": "active", "repos": [{"repo_id": "org/model"}]})
    preview = worker.preview(cache / "hub", [selector()])
    assert not preview["selected"][0]["commits"]
    assert "preparing" in preview["blockers"][0]
    with pytest.raises(worker.StorageError, match="preparing"):
        delete(cache, [selector()])
    assert repo.exists()


@pytest.mark.parametrize("reason", ["filesystem", "shared-blobs"])
def test_damaged_cache_without_commits_keeps_storage_safety_blockers(cache, monkeypatch, reason):
    repo = add_revision(cache)
    (repo / "refs/main").write_text(COMMIT_B)
    if reason == "filesystem":
        monkeypatch.setattr(worker, "filesystem", lambda hub: {"coordinated": False, "type": "nfs"})
    else:
        (cache / "hub/blobs").mkdir()
    preview = worker.preview(cache / "hub", [selector()])
    assert not preview["selected"][0]["commits"]
    assert preview["blockers"]
    with pytest.raises(worker.StorageError):
        delete(cache, [selector()])
    assert repo.exists()


@pytest.mark.parametrize("broken", [False, True])
def test_canonical_repo_name_cannot_be_redirected(cache, broken):
    repo = add_revision(cache)
    if broken:
        (repo / "refs/main").write_text(COMMIT_B)
    with pytest.raises(worker.StorageError, match="Invalid cache basename"):
        worker.selection(cache / "hub", [{**selector("other/model"), "cache_name": repo.name}])
    assert repo.exists()


def test_dangling_internal_file_links_can_be_removed_but_not_external_links(cache, tmp_path):
    repo = add_revision(cache)
    (repo / "blobs/weights").unlink()
    preview = worker.preview(cache / "hub", [selector()])
    assert not preview["blockers"]
    outside = repo / "external"
    outside.symlink_to(tmp_path / "missing-external")
    assert worker.preview(cache / "hub", [selector()])["blockers"]
    outside.unlink()
    assert delete(cache, [selector()])["state"] == "removed"


def test_non_utf8_ref_does_not_prevent_whole_repo_removal(cache):
    repo = add_revision(cache)
    (repo / "refs/main").write_bytes(b"\xff\xfe")
    preview = worker.preview(cache / "hub", [selector()])
    assert not preview["blockers"]
    assert preview["selected"][0]["cache_state"] == "corrupt-or-incomplete"
    delete(cache, [selector()], preview)
    assert not repo.exists()


def test_hardlinks_outside_selected_tree_reduce_estimated_reclamation(cache, tmp_path):
    import os

    repo = add_revision(cache, content=b"x" * 8192)
    os.link(repo / "blobs/weights", tmp_path / "external-hardlink")
    row = worker.preview(cache / "hub", [selector()])["selected"][0]
    assert row["allocated_bytes"] - row["reclaimable_bytes"] == (repo / "blobs/weights").stat().st_blocks * 512
    delete(cache, [selector()])
    assert (tmp_path / "external-hardlink").read_bytes() == b"x" * 8192


def test_repository_remove_verifies_exact_path_disappears(cache, monkeypatch):
    repo = add_revision(cache)

    def no_delete(path):
        pass

    no_delete.avoids_symlink_attacks = True
    monkeypatch.setattr(worker.shutil, "rmtree", no_delete)
    with pytest.raises(worker.StorageError, match="verification failed"):
        delete(cache, [selector()])
    assert repo.exists()


def test_special_file_and_symlink_roots_block_whole_repo_removal(cache):
    import os

    repo = add_revision(cache)
    os.mkfifo(repo / "refs/fifo")
    preview = worker.preview(cache / "hub", [selector()])
    assert "Special file" in preview["blockers"][0]
    (repo / "refs/fifo").unlink()
    alias = repo.parent / "models--org--alias"
    alias.symlink_to(repo, target_is_directory=True)
    assert "symlink" in worker.preview(cache / "hub", [selector("org/alias")])["blockers"][0]
    assert repo.exists()


def test_nested_bind_mount_blocks_even_on_same_device(cache, monkeypatch):
    repo = add_revision(cache)
    original = Path.read_text

    def read_text(path, *args, **kwargs):
        value = original(path, *args, **kwargs)
        if str(path) == "/proc/self/mountinfo":
            value += "999 1 0:1 / %s rw - ext4 fixture rw\n" % (repo / "blobs")
        return value

    monkeypatch.setattr(Path, "read_text", read_text)
    preview = worker.preview(cache / "hub", [selector()])
    assert "Mounted path" in preview["blockers"][0]
    assert preview["selected"][0]["allocated_bytes"] is None
    assert repo.exists()


@pytest.mark.parametrize("repo", ["../model", "/home/me", "x;touch pwn", "org/../model", "org/model:Q4", "models--org--model", "a/b/c"])
def test_invalid_selectors_rejected(repo):
    with pytest.raises(worker.StorageError):
        worker.validate_selector({"repo_id": repo})


def test_symlinked_cache_root_refused(tmp_path):
    (tmp_path / "real").mkdir()
    (tmp_path / "alias").symlink_to(tmp_path / "real", target_is_directory=True)
    with pytest.raises(worker.StorageError):
        worker.hub_path(str(tmp_path / "alias"))


def test_readonly_registry_does_not_migrate_or_create(tmp_path):
    from sparkrun.core.registry import RegistryManager

    root, cache = tmp_path / "config", tmp_path / "registry-cache"
    manager = RegistryManager(root, cache, read_only=True)
    manager.list_registries()
    manager.ensure_initialized()
    assert not root.exists()
    assert not cache.exists()


def test_model_inputs_keep_dependency_revisions_and_recipe_unmodified():
    from sparkrun.core.recipe import Recipe
    from sparkrun.core.model_preparation import model_inputs
    from sparkrun.runtimes.sglang import SglangRuntime

    recipe = Recipe.from_dict(
        {
            "name": "draft",
            "model": "org/main",
            "model_revision": COMMIT_A,
            "runtime": "sglang",
            "defaults": {"speculative_draft_model_path": "org/draft", "speculative_draft_model_revision": COMMIT_B},
        }
    )
    before = deepcopy_entries(recipe)
    inputs = model_inputs(recipe, SglangRuntime())
    assert [(entry.name, entry.revision) for entry in inputs] == [("org/main", COMMIT_A), ("org/draft", COMMIT_B)]
    assert deepcopy_entries(recipe) == before


def deepcopy_entries(recipe):
    return [replace(entry, target=list(entry.target)) for entry in recipe.distribution_config.models.entries]


@pytest.mark.parametrize("runtime_name", ["vllm-distributed", "vllm-ray", "atlas"])
def test_all_draft_runtimes_share_pure_dependencies_with_prepare(runtime_name):
    from sparkrun.core.recipe import Recipe
    from sparkrun.core.model_preparation import model_inputs
    from sparkrun.runtimes.vllm_distributed import VllmDistributedRuntime
    from sparkrun.runtimes.vllm_ray import VllmRayRuntime
    from sparkrun.runtimes.atlas import AtlasRuntime

    runtime = {"vllm-distributed": VllmDistributedRuntime, "vllm-ray": VllmRayRuntime, "atlas": AtlasRuntime}[runtime_name]()
    defaults = (
        {"draft_model": "org/draft", "draft_model_revision": COMMIT_B}
        if runtime_name == "atlas"
        else {"speculative_config": {"model": "org/draft", "revision": COMMIT_B}}
    )
    recipe = Recipe.from_dict(
        {"name": "draft-test", "model": "org/main", "model_revision": COMMIT_A, "runtime": runtime_name, "defaults": defaults}
    )
    before = deepcopy_entries(recipe)
    inputs = model_inputs(recipe, runtime)
    assert deepcopy_entries(recipe) == before
    runtime.prepare(recipe, [], dry_run=True)
    recipe.distribution_config.resolve(recipe)
    assert [(e.name, e.revision) for e in recipe.distribution_config.models.entries] == [(e.name, e.revision) for e in inputs]
    assert inputs[-1].revision == COMMIT_B and inputs[-1].name == "org/draft"
