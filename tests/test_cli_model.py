"""Storage command contract: command names, JSON, confirmation, exact scope."""

from dataclasses import replace
import json
from unittest.mock import Mock

from click.testing import CliRunner
import pytest

from sparkrun.api import model
from sparkrun.cli import main


@pytest.fixture
def preview():
    from sparkrun.api.model._types import RemovalTarget

    location = model.CacheLocation("node-a", "/cache/hf", "cluster")
    return model.RemovalPlan(
        (
            RemovalTarget(
                location,
                ({"repo_id": "org/model"},),
                {
                    "selected": [
                        {
                            "repo_id": "org/model",
                            "commits": ["a" * 40],
                            "reclaimable_bytes": 1234,
                            "allocated_bytes": 4096,
                            "path": "/cache/hf/hub/models--org--model",
                            "action": "remove-repository",
                            "cache_state": "cached",
                            "state": "selected",
                            "contents": {
                                "files": 3,
                                "blobs": 1,
                                "snapshots": [{"commit": "a" * 40, "files": ["Q8/weights.gguf"]}],
                                "refs": [],
                                "partial_files": [],
                            },
                        }
                    ],
                    "identity": "fixture",
                    "usage": {"models": [{"repo_id": "org/model", "state": "no-managed-consumers"}]},
                    "blockers": [],
                    "reclaimable_bytes": 1234,
                },
            ),
        )
    )


def test_help_lists_native_group_and_setup_command():
    runner = CliRunner()
    for command in (["model", "--help"], ["setup", "model-storage", "--help"]):
        result = runner.invoke(main, command)
        assert result.exit_code == 0, result.output
    assert "--yes" in runner.invoke(main, ["model", "remove", "--help"]).output
    from sparkrun.cli._model import model as commands

    assert set(commands.commands) == {"list", "show", "pull", "remove"}


def test_list_json(monkeypatch):
    monkeypatch.setattr(model, "list_cached", lambda options: model.InventoryReport(()))
    result = CliRunner().invoke(main, ["model", "list", "--local", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"schema_version": 1, "caches": [], "errors": [], "complete": True, "usage_scope": "managed"}


@pytest.mark.parametrize("name", ["ls", "rm"])
def test_removed_aliases_are_not_commands(name):
    result = CliRunner().invoke(main, ["model", name])
    assert result.exit_code == 2
    assert "No such command" in result.output


def test_dry_run_never_deletes_or_confirms(monkeypatch, preview):
    monkeypatch.setattr(model, "plan_remove", lambda *args: preview)
    remove = Mock(side_effect=AssertionError("must not delete"))
    monkeypatch.setattr(model, "remove", remove)
    result = CliRunner().invoke(main, ["model", "remove", "org/model", "--local", "--dry-run", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["targets"][0]["selected"][0]["repo_id"] == "org/model"
    assert not remove.called


def test_human_removal_plan_shows_both_conditions_and_exact_scopes(monkeypatch, preview):
    from copy import deepcopy

    first = preview.targets[0]
    damaged = deepcopy(first.snapshot)
    entry = damaged["selected"][0]
    entry["cache_state"] = "corrupt-or-incomplete"
    entry["commits"] = []
    entry["issues"] = ["Ref main -> missing has no snapshot directory"]
    plan = replace(preview, targets=(first, replace(first, location=replace(first.location, host="node-b"), snapshot=damaged)))
    monkeypatch.setattr(model, "plan_remove", lambda *args: plan)
    remove = Mock()
    monkeypatch.setattr(model, "remove", remove)
    result = CliRunner().invoke(main, ["model", "remove", "org/model", "--dry-run"])
    assert result.exit_code == 0, result.output
    for text in (
        "Host",
        "Disk usage",
        "Est. reclaimable",
        "Action",
        "Contents found",
        "Recognized metadata",
        "Inconsistent cache",
        "node-a",
        "node-b",
        "Remove entire model cache",
        "4.0 KiB",
        "Q8",
        "a" * 40,
        "Ref main -> missing",
        "/cache/hf/hub/models--org--model",
        "Recipes, container images",
        "partial downloads",
    ):
        assert text in result.output
    assert "already-absent" not in result.output
    assert not remove.called


def test_blocked_host_is_visible_with_nothing_deleted_notice(monkeypatch, preview):
    from copy import deepcopy

    target = preview.targets[0]
    snapshot = deepcopy(target.snapshot)
    snapshot["selected"][0].update(action="blocked", state="blocked", error="Unsafe path", reclaimable_bytes=None)
    plan = replace(preview, targets=(replace(target, snapshot=snapshot),), blockers=("node-a: Unsafe path",))
    monkeypatch.setattr(model, "plan_remove", lambda *args: plan)
    result = CliRunner().invoke(main, ["model", "remove", "org/model", "--yes"])
    assert result.exit_code == 1
    assert "node-a" in result.output and "Unsafe path" in result.output
    assert "Nothing deleted on any host" in result.output


def test_confirmation_names_the_number_of_hosts_and_caches(monkeypatch, preview):
    import sys

    def plan(*args):
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        return preview

    monkeypatch.setattr(model, "plan_remove", plan)
    remove = Mock(return_value=model.OperationResult(({"state": "removed"},)))
    monkeypatch.setattr(model, "remove", remove)
    result = CliRunner().invoke(main, ["model", "remove", "org/model"], input="n\n")
    assert result.exit_code == 1, result.output
    assert "Apply this removal plan to 1 cache(s) on 1 host(s)?" in result.output
    assert not remove.called
    result = CliRunner().invoke(main, ["model", "remove", "org/model"], input="y\n")
    assert result.exit_code == 0, result.output
    assert remove.called


def test_noninteractive_requires_yes(monkeypatch, preview):
    monkeypatch.setattr(model, "plan_remove", lambda *args: preview)
    remove = Mock()
    monkeypatch.setattr(model, "remove", remove)
    result = CliRunner().invoke(main, ["model", "remove", "org/model", "--local"])
    assert result.exit_code == 2
    assert "requires --yes" in result.output
    assert not remove.called


def test_yes_never_bypasses_blockers(monkeypatch, preview):
    monkeypatch.setattr(model, "plan_remove", lambda *args: replace(preview, blockers=("Active workload",)))
    remove = Mock()
    monkeypatch.setattr(model, "remove", remove)
    result = CliRunner().invoke(main, ["model", "remove", "org/model", "--yes", "--json"])
    assert result.exit_code == 1
    assert json.loads(result.output)["blockers"] == ["Active workload"]
    assert not remove.called


def test_json_errors_and_target_validation():
    result = CliRunner().invoke(main, ["model", "remove", "org/model", "--recipe", "sample", "--json", "--dry-run"])
    assert result.exit_code == 2, result.output
    assert json.loads(result.output)["errors"][0]["code"] == "InvalidModelSelection"


def test_partial_removal_exit_nonzero(monkeypatch, preview):
    monkeypatch.setattr(model, "plan_remove", lambda *args: preview)
    monkeypatch.setattr(model, "remove", lambda *args: model.OperationResult(({"state": "indeterminate", "error": "SSH lost"},)))
    result = CliRunner().invoke(main, ["model", "remove", "org/model", "--yes", "--json"])
    assert result.exit_code == 1
    assert not json.loads(result.output)["complete"]


def test_pull_dry_run_does_not_download(monkeypatch):
    plan = model.PullPlan(
        (model.CacheLocation("node-a", "/hf", "cluster"),),
        ({"model": "org/model", "revision": "a" * 40, "target": [0]},),
        "local",
        "/control/hf",
    )
    monkeypatch.setattr(model, "plan_pull", lambda *args: plan)
    pull = Mock()
    monkeypatch.setattr(model, "pull", pull)
    result = CliRunner().invoke(main, ["model", "pull", "org/model", "--dry-run", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["staging_cache"] == "/control/hf"
    assert not pull.called


@pytest.fixture
def inventory():
    from copy import deepcopy

    cache = {
        "location": {"host": "node-a", "hub": "/hf/hub", "source": "cluster"},
        "filesystem": {"free_bytes": 2**40},
        "usage": {"state": "unknown", "coverage": "test coverage", "uncertainty": ["Cannot inspect native process 123"]},
        "receipts": [{"operation": "test-op", "recipe": "sample", "repos": [{"repo_id": "org/z-large"}]}],
        "warnings": ["test cache warning"],
        "models": [
            {
                "repo_id": "org/z-large",
                "attributed_bytes": 2**30,
                "state": "cached",
                "path": "/hf/hub/models--org--z-large",
                "revisions": [
                    {"commit": "a" * 40, "refs": ["main"], "attributed_bytes": 2**29, "files": ["config.json"]},
                    {"commit": "b" * 40, "refs": [], "attributed_bytes": 2**29, "files": ["config.json"]},
                ],
            },
            {"repo_id": "org/a-broken", "attributed_bytes": None, "state": "corrupt-or-incomplete", "revisions": []},
        ],
    }
    second = deepcopy(cache)
    second["location"]["host"] = "node-b"
    second["models"] = []
    return model.InventoryReport((cache, second))


def test_inventory_tables_preserve_hosts_warnings_and_sorting(monkeypatch, inventory):
    monkeypatch.setattr(model, "list_cached", lambda options: inventory)
    runner = CliRunner()
    result = runner.invoke(main, ["model", "list"])
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    header = next(line for line in lines if "Attributed size" in line)
    assert all(title in header for title in ["Model", "Revisions", "Cache state", "Origin"])
    assert any(line.strip() and set(line.strip()) == {"-"} for line in lines)
    broken = next(line for line in lines if "org/a-broken" in line)
    large = next(line for line in lines if "org/z-large" in line)
    assert "unknown" in broken and "corrupt-or-incomplete" in broken
    assert "1.0 GiB" in large and "receipt recorded" in large
    assert broken.index("corrupt-or-incomplete") == header.index("Cache state") == large.index("cached")
    assert result.output.index("org/a-broken") < result.output.index("org/z-large")
    assert "node-a: /hf/hub" in result.output and "node-b: /hf/hub" in result.output
    assert "No cached models found." in result.output
    assert "Cannot inspect native process 123" in result.output and "test cache warning" in result.output
    by_size = runner.invoke(main, ["model", "list", "--sort", "size"])
    assert by_size.exit_code == 0, by_size.output
    assert by_size.output.index("org/z-large") < by_size.output.index("org/a-broken")


def test_inventory_revisions_are_tabulated_without_truncating(monkeypatch, inventory):
    monkeypatch.setattr(model, "list_cached", lambda options: inventory)
    result = CliRunner().invoke(main, ["model", "list", "--revisions"], terminal_width=40)
    assert result.exit_code == 0, result.output
    assert "Commit" in result.output and "Refs" in result.output
    assert "a" * 40 in result.output and "b" * 40 in result.output
    assert "main" in result.output
    assert result.output.count("org/z-large") == 2
    assert "org/a-broken" in result.output


def test_size_sort_uses_physical_bytes_for_damaged_caches(monkeypatch, inventory):
    inventory.caches[0]["models"][1]["allocated_bytes"] = 2**35
    monkeypatch.setattr(model, "list_cached", lambda options: inventory)
    result = CliRunner().invoke(main, ["model", "list", "--sort", "size"])
    assert result.exit_code == 0, result.output
    assert "Disk usage" in result.output and "32.0 GiB" in result.output
    assert result.output.index("org/a-broken") < result.output.index("org/z-large")


def test_inventory_json_unchanged_and_partial_errors_visible(monkeypatch, inventory):
    report = replace(inventory, errors=({"message": "node-c unreachable", "code": "StorageUnreachable"},))
    monkeypatch.setattr(model, "list_cached", lambda options: report)
    runner = CliRunner()
    result = runner.invoke(main, ["model", "list", "--json", "--revisions"])
    assert result.exit_code == 1
    assert json.loads(result.output) == json.loads(json.dumps(report.to_dict()))
    text = runner.invoke(main, ["model", "list"])
    assert text.exit_code == 1
    assert "Attributed size" in text.output and "Error: node-c unreachable" in text.output


def test_show_keeps_detailed_files_and_receipts(monkeypatch, inventory):
    cache = {**inventory.caches[0], "models": [inventory.caches[0]["models"][0]]}
    monkeypatch.setattr(model, "inspect", lambda *args: model.ModelReport((cache,)))
    result = CliRunner().invoke(main, ["model", "show", "org/z-large"])
    assert result.exit_code == 0, result.output
    assert "config.json" in result.output and "Path: /hf/hub/models--org--z-large" in result.output
    assert "Receipt: recipe=sample operation=test-op" in result.output
