"""Thin renderers over the model storage API. Confirmation lives only here."""

from __future__ import annotations

from functools import wraps
import json
import sys

import click

from ._common import host_options


@click.group()
def model():
    """List, inspect, prefetch and remove cached model weights.

    Recipes, serving workloads, container images and weights have separate
    lifecycles. Stop/unload a model before removing its retained weights.
    """


def _options(function):
    function = host_options(function)
    function = click.option("--local", is_flag=True, help="Use only the control machine, ignoring default hosts")(function)
    function = click.option("--include-local", is_flag=True, help="Also inspect/remove the control-machine copy")(function)
    function = click.option("--cache-dir", help="HF home on targets (the Hub cache is its hub/ subdirectory)")(function)
    function = click.option("--local-cache-dir", help="Separate control-machine HF home")(function)
    function = click.option("--json", "output_json", is_flag=True, help="Emit schema-versioned JSON")(function)
    return function


def _selector_options(function):
    function = click.argument("model_id", required=False)(function)
    function = click.option("--recipe", "recipe_name", help="Resolve a cached recipe explicitly (no registry refresh)")(function)
    function = click.option("--revision", help="Cached ref/commit for removal; Hub ref/commit for pull")(function)
    return function


def _errors(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        from sparkrun.api import SparkrunError
        from sparkrun.api.model import InvalidModelSelection

        try:
            return function(*args, **kwargs)
        except SparkrunError as error:
            if kwargs.get("output_json"):
                click.echo(
                    json.dumps({"schema_version": 1, "complete": False, "errors": [{"code": type(error).__name__, "message": str(error)}]})
                )
                raise click.exceptions.Exit(2 if isinstance(error, InvalidModelSelection) else 1) from error
            if isinstance(error, InvalidModelSelection):
                raise click.UsageError(str(error)) from error
            raise click.ClickException(str(error)) from error
        except KeyboardInterrupt as error:
            raise click.exceptions.Exit(130) from error

    return wrapped


def _storage_options(kwargs, **extra):
    from sparkrun.api.model import StorageOptions

    hosts = kwargs.pop("hosts")
    return StorageOptions(
        cluster=kwargs.pop("cluster_name"),
        hosts=tuple(h.strip() for h in hosts.split(",") if h.strip()) if hosts is not None else None,
        hosts_file=kwargs.pop("hosts_file"),
        local=kwargs.pop("local"),
        include_local=kwargs.pop("include_local"),
        cache_dir=kwargs.pop("cache_dir"),
        local_cache_dir=kwargs.pop("local_cache_dir"),
        **extra,
    )


def _target(model_id, recipe_name, revision, include_dependencies=False):
    from sparkrun.api.model import ModelSelector

    return ModelSelector(model=model_id, recipe=recipe_name, revision=revision, include_dependencies=include_dependencies)


def _size(value):
    if value is None:
        return "unknown"
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return "%.1f %s" % (value, unit)
        value /= 1024
    return "unknown"


def _location(location):
    click.echo("%s: %s [%s]" % (location["host"], location["hub"], location["source"]))


def _inventory(report, *, revisions=False, sort="name", details=False):
    from sparkrun.utils.cli_formatters import format_model_storage_table

    for index, cache in enumerate(report.caches):
        if index:
            click.echo()
        _location(cache["location"])
        click.echo("  Free: %s; usage: %s" % (_size(cache["filesystem"]["free_bytes"]), cache["usage"]["state"]))
        click.echo("  Coverage: " + cache["usage"]["coverage"])
        for uncertainty in cache["usage"]["uncertainty"]:
            click.echo("  Warning: " + uncertainty)
        for observation in cache["usage"].get("managed_observations", ()):
            click.echo("  Usage: " + observation)
        models = sorted(
            cache["models"],
            key=(lambda m: -(m.get("allocated_bytes") if m.get("allocated_bytes") is not None else m["attributed_bytes"] or 0))
            if sort == "size"
            else lambda m: m["repo_id"],
        )
        if not details:
            recorded = {p.get("repo_id") for receipt in cache["receipts"] for p in receipt.get("repos", [])}
            rows = []
            for entry in models:
                row = {
                    "model": entry["repo_id"],
                    "size": _size(entry["attributed_bytes"]),
                    "disk": _size(entry.get("allocated_bytes")),
                    "revisions": str(len(entry["revisions"])),
                    "state": entry["state"],
                    "usage": entry.get("usage", {}).get("state", "unknown"),
                    "origin": "receipt recorded" if entry["repo_id"] in recorded else "unknown",
                }
                if revisions:
                    for rev in entry["revisions"] or [None]:
                        rows.append({**row, "commit": rev["commit"] if rev else "-", "refs": ", ".join(rev["refs"]) or "-" if rev else "-"})
                else:
                    rows.append(row)
            click.echo(format_model_storage_table(rows, show_revisions=revisions))
        elif not models:
            click.echo("  No cached models found.")
        for entry in models if details else ():
            receipts = [r for r in cache["receipts"] if any(p.get("repo_id") == entry["repo_id"] for p in r.get("repos", []))]
            click.echo(
                "  %s  %s attributed  %d recognized revision(s)  %s"
                % (
                    entry["repo_id"],
                    _size(entry["attributed_bytes"]),
                    len(entry["revisions"]),
                    "receipt recorded" if receipts else "origin unknown",
                )
            )
            if entry["state"] != "cached":
                click.echo("    Warning: " + entry["state"])
            click.echo("    Usage: " + entry.get("usage", {}).get("state", "unknown"))
            if revisions or details:
                for revision in entry["revisions"]:
                    click.echo(
                        "    %s  refs=%s  %s attributed"
                        % (revision["commit"], ",".join(revision["refs"]) or "-", _size(revision["attributed_bytes"]))
                    )
                    if details:
                        for filename in revision["files"]:
                            click.echo("      " + filename)
            if details:
                click.echo("    Path: " + entry["path"])
                click.echo("    Disk usage: " + _size(entry.get("allocated_bytes")))
                _contents(entry, indent="    ", files=not entry["revisions"])
                for receipt in receipts:
                    click.echo("    Receipt: recipe=%s operation=%s" % (receipt.get("recipe") or "unknown", receipt["operation"]))
        for entry in models:
            usage = entry.get("usage", {})
            for job in usage.get("jobs", []):
                click.echo("  Using %s: job %s (recipe %s)" % (entry["repo_id"], job["job"], job.get("recipe") or "unknown"))
            for reservation in usage.get("reservations", []):
                click.echo("  Preparing %s: operation %s" % (entry["repo_id"], reservation.get("operation", "unknown")))
        for warning in cache["warnings"]:
            click.echo("  Warning: " + warning)
    for error in report.errors:
        click.echo("Error: " + error["message"], err=True)
    for path in getattr(report, "externally_managed", ()):
        click.echo("Externally managed: " + path)


def _contents(entry, *, indent="  ", files=False):
    contents = entry.get("contents", {})
    if contents:
        click.echo(indent + "Repository entries: %d files/links, including %d blob entries." % (contents["files"], contents["blobs"]))
    for snapshot in contents.get("snapshots", []):
        click.echo(indent + "Snapshot directory: " + snapshot["commit"] + " (%d file entries)" % len(snapshot["files"]))
        if files:
            for name in snapshot["files"]:
                click.echo(indent + "  " + name)
    variants = sorted(
        {
            name.rsplit("/", 1)[0] if "/" in name else name
            for s in contents.get("snapshots", [])
            for name in s["files"]
            if name.lower().endswith(".gguf")
        }
    )
    if variants:
        click.echo(indent + "GGUF paths/variants: " + ", ".join(variants))
    for ref in contents.get("refs", []):
        click.echo(indent + "Ref: %s -> %s" % (ref["name"], ref["commit"]))
    for name in contents.get("partial_files", []):
        click.echo(indent + "Partial download: " + name)
    for issue in entry.get("issues", []):
        click.echo(indent + "Warning: " + issue)
    if contents:
        click.echo(indent + "Contents are filesystem observations, not a completeness/integrity check.")


def _action_label(entry):
    return {
        "remove-repository": "Remove entire model cache",
        "remove-revisions": "Remove selected revisions",
        "already-absent": "Nothing to remove",
        "blocked": "Blocked",
    }.get(entry.get("action"), "Remove selected revisions")


def _render_plan(plan):
    from sparkrun.utils.cli_formatters import format_model_removal_table

    click.echo("Removal scope: entire model cache, or explicitly selected revisions; includes ALL GGUF variants in that scope.")
    click.echo("Usage scope: SparkRun-managed jobs and preparations only; external clients are not covered.")
    repos = sorted({e["repo_id"] for t in plan.targets for e in t.snapshot["selected"]})
    for repo in repos:
        click.echo("\nModel: " + repo)
        rows, details = [], []
        for target in plan.targets:
            for entry in target.snapshot["selected"]:
                if entry["repo_id"] != repo:
                    continue
                contents = entry.get("contents", {})
                snapshots = contents.get("snapshots", [])
                ggufs = sum(name.lower().endswith(".gguf") for s in snapshots for name in s["files"])
                snapshot_files = sum(len(s["files"]) for s in snapshots)
                usage = next((m["state"] for m in target.snapshot.get("usage", {}).get("models", []) if m["repo_id"] == repo), "unknown")
                rows.append(
                    {
                        "host": target.location.host,
                        "condition": {"cached": "Recognized metadata", "corrupt-or-incomplete": "Inconsistent cache"}.get(
                            entry.get("cache_state"), entry.get("cache_state", "unknown")
                        ),
                        "contents": "%s / %d partial"
                        % ("%d GGUF" % ggufs if ggufs else "%d snapshot files" % snapshot_files, len(contents.get("partial_files", [])))
                        if contents
                        else "unknown"
                        if entry.get("state") != "already-absent"
                        else "none",
                        "disk": _size(entry.get("allocated_bytes")),
                        "reclaim": _size(entry.get("reclaimable_bytes")),
                        "action": "Blocked"
                        if entry.get("state") != "already-absent"
                        and (target.snapshot.get("blockers") or usage in {"in-use", "preparing", "unknown"})
                        else _action_label(entry),
                        "usage": usage if entry.get("state") != "already-absent" else "not applicable",
                    }
                )
                details.append((target, entry))
        click.echo(format_model_removal_table(rows))
        for target, entry in details:
            click.echo("\n  %s: %s" % (target.location.host, entry.get("path", target.location.hub + " (not inspected)")))
            if entry.get("action") == "remove-repository":
                click.echo("    Deletes this entire directory: weights, snapshots, refs, metadata, orphan blobs and partial downloads.")
            elif entry.get("action") == "remove-revisions":
                click.echo("    Selected revisions: " + ", ".join(entry["commits"]))
                click.echo("    Deletes selected snapshots/refs and their unshared blobs; other repository files remain.")
            _contents(entry, indent="    ")
            if entry.get("error"):
                click.echo("    Blocked: " + entry["error"])
    sizes = [e.get("reclaimable_bytes") for t in plan.targets for e in t.snapshot["selected"]]
    total = sum(sizes) if all(s is not None for s in sizes) else None
    click.echo("\nEstimated reclaimable across selected caches: %s (actual freed space may differ)." % _size(total))
    click.echo("Recipes, container images and SparkRun coordination records are kept.")
    for blocker in plan.blockers:
        click.echo("Blocked: " + blocker)
    if plan.blockers:
        click.echo("Nothing deleted on any host: removal is blocked.")


def _result(result, output_json):
    if output_json:
        click.echo(json.dumps(result.to_dict()))
    else:
        for row in result.results:
            click.echo(
                "%s%s: %s%s"
                % (
                    row.get("location", {}).get("host", ""),
                    " " + row["model"] if "model" in row else "",
                    row["state"],
                    " — " + row["error"] if row.get("error") else "",
                )
            )
    if not result.complete:
        raise click.exceptions.Exit(1)


@model.command("list")
@_options
@click.option("--revisions", is_flag=True, help="Include cached commit hashes and refs")
@click.option("--sort", type=click.Choice(["size", "name"]), default="name", show_default=True)
@_errors
def model_list(output_json, revisions, sort, **kwargs):
    """List downloaded models, including weights no longer referenced by recipes."""
    from sparkrun.api import model as api

    report = api.list_cached(_storage_options(kwargs))
    if output_json:
        click.echo(json.dumps(report.to_dict()))
    else:
        _inventory(report, revisions=revisions, sort=sort)
    if not report.complete:
        raise click.exceptions.Exit(1)


@model.command("show")
@_options
@_selector_options
@click.option("--include-dependencies", is_flag=True)
@_errors
def model_show(model_id, recipe_name, revision, include_dependencies, output_json, **kwargs):
    """Inspect cached paths, revisions, files, usage and preparation receipts."""
    from sparkrun.api import model as api

    report = api.inspect(_target(model_id, recipe_name, revision, include_dependencies), _storage_options(kwargs))
    if output_json:
        click.echo(json.dumps(report.to_dict()))
    else:
        _inventory(report, details=True)
    if not report.complete:
        raise click.exceptions.Exit(1)


@model.command("remove")
@_options
@_selector_options
@click.option("--include-dependencies", is_flag=True, help="Also select recipe draft/other distributed model inputs")
@click.option("--dry-run", is_flag=True, help="Read-only real probes and removal preview; installs/deletes nothing")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation, never safety checks")
@_errors
def model_remove(model_id, recipe_name, revision, include_dependencies, dry_run, yes, output_json, **kwargs):
    """Remove an entire model cache (including incomplete files), or selected revisions. Does not stop workloads."""
    from sparkrun.api import model as api

    plan = api.plan_remove(_target(model_id, recipe_name, revision, include_dependencies), _storage_options(kwargs))
    if not output_json:
        _render_plan(plan)
    if dry_run or plan.blockers:
        if output_json:
            click.echo(json.dumps(plan.to_dict()))
        if plan.blockers:
            raise click.exceptions.Exit(1)
        return
    if not yes:
        if output_json or not sys.stdin.isatty():
            raise api.InvalidModelSelection("Noninteractive removal requires --yes (or preview with --dry-run)")
        click.confirm(
            "Apply this removal plan to %d cache(s) on %d host(s)?" % (len(plan.targets), len({t.location.host for t in plan.targets})),
            abort=True,
        )
    _result(api.remove(plan), output_json)


@model.command("pull")
@_options
@_selector_options
@click.option("--transfer-mode", type=click.Choice(["auto", "local", "push", "delegated", "pull"]), default="auto")
@click.option("--dry-run", is_flag=True, help="Resolve and show downloads without installing or transferring anything")
@_errors
def model_pull(model_id, recipe_name, revision, transfer_mode, dry_run, output_json, **kwargs):
    """Prefetch weights and recipe dependencies, without GPUs, images or serving."""
    from sparkrun.api import model as api

    plan = api.plan_pull(_target(model_id, recipe_name, revision), _storage_options(kwargs, transfer_mode=transfer_mode))
    if dry_run:
        if output_json:
            click.echo(json.dumps(plan.to_dict()))
        else:
            for loc in plan.locations:
                _location(loc.to_dict())
            click.echo("Transfer: %s; stages selected hosts independently of GPU occupancy" % plan.transfer_mode)
            if plan.transfer_mode in {"local", "push"}:
                click.echo("Control staging cache: " + plan.local_cache_dir)
            for item in plan.inputs:
                click.echo("  %s @ %s -> node indices %s" % (item["model"], item["revision"] or "externally-managed", item["target"]))
        return
    _result(api.pull(plan), output_json)


@click.command("model-storage")
@_options
@click.option("--offline", is_flag=True, help="Install using cached wheels and an existing uv binary only")
@_errors
def setup_model_storage(output_json, offline, **kwargs):
    """Prepare the pinned private model cache helper (no global hf CLI needed)."""
    from sparkrun.api import model as api

    _result(api.setup_storage(_storage_options(kwargs), offline=offline), output_json)
