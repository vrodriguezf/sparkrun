"""Model inventory, prefetch and guarded removal. No console or GPU scheduling."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from copy import copy, deepcopy
from dataclasses import replace
from pathlib import Path
from functools import wraps
import uuid

from sparkrun.api._context import resolve_sctx
from sparkrun.api._errors import HostsUnreachable, SparkrunError
from sparkrun.api.model import _errors
from sparkrun.api.model._errors import InvalidModelSelection, ModelInUse, StorageCapabilityMissing
from sparkrun.api.model._types import (
    CacheLocation,
    InventoryReport,
    ModelReport,
    ModelSelector,
    OperationResult,
    PullPlan,
    RemovalPlan,
    RemovalTarget,
    StorageOptions,
)
from sparkrun.models.storage_remote import WorkerError, WorkerSession, query, setup_helper


def _translate(error):
    cls = getattr(_errors, getattr(error, "code", "StorageUnreachable"), _errors.StorageUnreachable)
    return cls(str(error))


def _storage_errors(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        from sparkrun.core.recipe import RecipeError
        from sparkrun.core.cluster_manager import ClusterError
        from sparkrun.core.hosts import HostResolutionError

        try:
            return function(*args, **kwargs)
        except SparkrunError:
            raise
        except WorkerError as error:
            raise _translate(error) from error
        except (OSError, ValueError, RecipeError, ClusterError, HostResolutionError) as error:
            raise SparkrunError(str(error)) from error

    return wrapped


def _context(sctx):
    """Use the application's binding, but never bootstrap/migrate its catalogs."""
    from sparkrun.core.cluster_manager import ClusterManager
    from sparkrun.core.registry import RegistryManager

    sctx = copy(resolve_sctx(sctx))
    root = sctx.config.config_path.parent
    sctx.cluster_manager = ClusterManager(root, read_only=True)
    sctx.registry_manager = RegistryManager(root, sctx.config.cache_dir / "registries", read_only=True)
    return sctx


def _recipe(selector, sctx):
    from sparkrun.core.recipe import Recipe, discover_cwd_recipes
    from sparkrun.api._resolve import resolve_recipe

    if bool(selector.model) == bool(selector.recipe):
        raise InvalidModelSelection("Specify exactly one MODEL or --recipe")
    if selector.recipe is None:
        return None
    if selector.revision is not None:
        raise InvalidModelSelection("--revision applies to MODEL; recipe entries carry their own pins")
    if isinstance(selector.recipe, Recipe):
        recipe = deepcopy(selector.recipe)
    else:
        try:
            recipe = resolve_recipe(selector.recipe, sctx=sctx, local_files=discover_cwd_recipes(), offline=True)
        except Exception as error:
            raise InvalidModelSelection("Cannot resolve cached recipe %r: %s" % (selector.recipe, error)) from error
    recipe.resolve({})
    return recipe


def _selectors(selector, sctx, *, removal=False, pull=False):
    from sparkrun.api._resolve import resolve_runtime
    from sparkrun.core.model_preparation import model_inputs
    from sparkrun.core.recipe import DistributionModelEntry, is_local_model_path
    from sparkrun.models.download import model_cache_path, parse_gguf_model_spec
    from sparkrun.models.storage_worker import validate_selector

    recipe = _recipe(selector, sctx)
    if recipe is None:
        entries = [DistributionModelEntry(selector.model, revision=selector.revision)]
    else:
        entries = list(model_inputs(recipe, resolve_runtime(recipe, sctx=sctx)))
        primary = next(
            (entry for entry in entries if entry.name == recipe.model), DistributionModelEntry(recipe.model, revision=recipe.model_revision)
        )
        if not pull and not selector.include_dependencies:
            entries = [primary]
        elif all(entry.name != primary.name for entry in entries):
            entries.insert(0, primary)
        if recipe.cluster_config and recipe.cluster_config.resolved_model_path:
            entries = [DistributionModelEntry(recipe.cluster_config.resolved_model_path)]
    result = []
    for entry in entries:
        if is_local_model_path(entry.name):
            if removal:
                raise InvalidModelSelection("Pre-placed model paths are externally managed and cannot be removed")
            result.append(
                {"model": entry.name, "repo_id": None, "revision": None, "target": list(entry.target), "state": "externally-managed"}
            )
            continue
        repo_id, quant = parse_gguf_model_spec(entry.name)
        if quant and removal:
            raise InvalidModelSelection(
                "Selective GGUF removal is not supported. To remove ALL variants, explicitly use: model remove %s" % repo_id
            )
        try:
            validate_selector({"repo_id": repo_id, "revision": entry.revision})
        except Exception as error:
            raise InvalidModelSelection(str(error)) from error
        row = {
            "model": entry.name,
            "repo_id": repo_id,
            "revision": entry.revision,
            "cache_name": Path(model_cache_path(repo_id, "/__cache_home__")).name,
            "target": list(entry.target),
        }
        if row not in result:
            result.append(row)
    return tuple(result), recipe


def _locations(options, sctx, recipe=None):
    from sparkrun.api._resolve import resolve_cluster
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.hosts import parse_hosts_file
    from sparkrun.core.launcher import _enforce_recipe_mount_trust, resolve_effective_cache_dir, resolve_recipe_trust
    from sparkrun.orchestration.primitives import build_ssh_kwargs
    from sparkrun.orchestration.ssh import should_run_locally

    if options.local and (options.cluster is not None or options.hosts is not None or options.hosts_file):
        raise InvalidModelSelection("--local cannot be combined with --cluster, --hosts or --hosts-file")
    if options.hosts is not None and options.hosts_file:
        raise InvalidModelSelection("Specify --hosts or --hosts-file, not both")
    hosts = parse_hosts_file(options.hosts_file) if options.hosts_file else options.hosts
    defaulted = options.cluster is None and hosts is None and not options.local
    if options.local:
        cluster = ClusterDefinition(name="", hosts=["localhost"])
    else:
        selected_cluster = options.cluster
        if defaulted:
            # Unlike launch's best-effort defaults, a storage command must not
            # silently change scope to local when a configured cluster is broken.
            selected_cluster = sctx.cluster_manager.get_default()
        try:
            cluster = resolve_cluster(selected_cluster, hosts, sctx=sctx)
        except HostsUnreachable:
            if not defaulted or selected_cluster is not None:
                raise
            cluster = ClusterDefinition(name="", hosts=["localhost"])
    if not cluster.hosts:
        raise InvalidModelSelection("Storage requires at least one host")
    if cluster.transport != "ssh" or cluster.executor not in {None, "docker", "local"}:
        raise StorageCapabilityMissing("This provider needs a model storage adapter; its control cache is not its model volume")
    config = sctx.config if options.local else sctx.config.for_cluster(cluster)
    ssh = {} if options.local else build_ssh_kwargs(config)
    cache_dir = options.cache_dir or (options.local_cache_dir if options.local else None) or cluster.cache_dir
    local_cache = options.local_cache_dir or str(config.hf_cache_dir)
    if recipe:
        _enforce_recipe_mount_trust(recipe, resolve_recipe_trust(recipe, False, sctx=sctx))
        if recipe.cluster_config:
            cache_dir = (recipe.cluster_config.local_cache_dir if options.local else recipe.cluster_config.remote_cache_dir) or cache_dir
            local_cache = recipe.cluster_config.local_cache_dir or local_cache
    cache_dir = resolve_effective_cache_dir(cache_dir, list(cluster.hosts), ssh, config)
    source = "recipe/option/cluster" if options.cache_dir or cluster.cache_dir or (recipe and recipe.cluster_config) else "host default"
    if defaulted:
        source += " (default selection)"
    locations = []
    for host in cluster.hosts:
        host = "localhost" if should_run_locally(host, ssh.get("ssh_user")) else host
        loc = CacheLocation(host, str(cache_dir), source, ssh, cluster)
        if loc not in locations:
            locations.append(loc)
    if options.include_local:
        loc = CacheLocation("localhost", str(local_cache), "control-machine copy", {})
        if not any(existing.host == loc.host and existing.hf_home == loc.hf_home for existing in locations):
            locations.append(loc)
    return tuple(locations), cluster, local_cache


def _query(location, action, **kwargs):
    return query(location.host, location.hf_home, action, ssh_kwargs=location.ssh_kwargs, **kwargs)


def _managed_usage(location, cache, sctx, selections):
    """Match live executor instances to target-side model bindings."""
    from sparkrun.api._status import status
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.cluster_status import ClusterStatus
    from sparkrun.core.model_usage import managed_usage
    from sparkrun.core.pending_ops import list_pending_ops

    try:
        snapshot = status(
            [location.host],
            cluster=location.cluster or ClusterDefinition(name="", hosts=[location.host]),
            sctx=sctx,
            ssh_kwargs={"ssh_user": None, "ssh_key": None, "ssh_options": [], **location.ssh_kwargs},
            read_only=True,
        )
        # Native jobs may use recipe/CLI-specific PID namespaces. Persisted
        # launch targets are read on the cache host, not only on the controller
        # that launched them. Probe these even when the default namespace is idle.
        roots = set()
        snapshots = [snapshot]
        for binding in cache.get("bindings", []):
            target = binding.get("target") or {}
            if target.get("executor") == "local":
                root = target["pid_dir"]
                if not isinstance(root, str) or not root:
                    raise ValueError("Invalid native storage binding target")
                if root not in roots:
                    roots.add(root)
                    snapshots.append(
                        status(
                            [location.host],
                            cluster=ClusterDefinition(name="", hosts=[location.host], executor="local", executor_config={"pid_dir": root}),
                            sctx=sctx,
                            ssh_kwargs={"ssh_user": None, "ssh_key": None, "ssh_options": [], **location.ssh_kwargs},
                            read_only=True,
                        )
                    )
        if len(snapshots) > 1:
            snapshot = replace(ClusterStatus.merge(snapshots), errors={h: e for s in snapshots for h, e in s.observation_errors.items()})
    except Exception as error:
        snapshot = ClusterStatus(errors={location.host: "Managed workload observation unavailable (%s)" % type(error).__name__})
    pending = [p for p in list_pending_ops(str(sctx.config.cache_dir), read_only=True) if not p.get("hosts") or location.host in p["hosts"]]
    return managed_usage(snapshot, location.host, cache, selections, pending)


def _inventory(locations):
    def probe(loc):
        try:
            return {"location": loc.to_dict(), **_query(loc, "scan")}, None
        except (WorkerError, OSError) as error:
            return None, {"location": loc.to_dict(), "code": getattr(error, "code", "StorageUnreachable"), "message": str(error)}

    with ThreadPoolExecutor(max_workers=min(8, len(locations))) as pool:
        observations = list(pool.map(probe, locations))
    caches, errors, seen = [], [], set()
    for cache, error in observations:
        if error:
            errors.append(error)
        elif cache is not None and cache["identity"] not in seen:
            seen.add(cache["identity"])
            caches.append(cache)
    return InventoryReport(tuple(caches), tuple(errors))


def _with_managed_usage(report, locations, sctx):
    by_path = {(loc.host, loc.hub): loc for loc in locations}
    caches = []
    for cache in report.caches:
        location = by_path[(cache["location"]["host"], cache["location"]["hub"])]
        selections = [
            {"repo_id": m["repo_id"], "commits": [r["commit"] for r in m["revisions"]], "action": "remove-repository"}
            for m in cache["models"]
        ]
        usage = _managed_usage(location, cache, sctx, selections)
        by_repo = {m["repo_id"]: m for m in usage["models"]}
        caches.append({**cache, "usage": usage, "models": [{**m, "usage": by_repo[m["repo_id"]]} for m in cache["models"]]})
    return replace(report, caches=tuple(caches))


@_storage_errors
def list_cached(options: StorageOptions | None = None, *, sctx=None) -> InventoryReport:
    sctx = _context(sctx)
    locations, _, _ = _locations(options or StorageOptions(), sctx)
    return _with_managed_usage(_inventory(locations), locations, sctx)


@_storage_errors
def inspect(selector: ModelSelector, options: StorageOptions | None = None, *, sctx=None) -> ModelReport:
    sctx = _context(sctx)
    selectors, recipe = _selectors(selector, sctx)
    locations, _, _ = _locations(options or StorageOptions(), sctx, recipe)
    report = _with_managed_usage(_inventory(locations), locations, sctx)
    ids = {s["repo_id"] for s in selectors}
    caches = tuple({**cache, "models": [m for m in cache["models"] if m["repo_id"] in ids]} for cache in report.caches)
    return ModelReport(
        caches,
        report.errors,
        selectors=selectors,
        externally_managed=tuple(s["model"] for s in selectors if s.get("state") == "externally-managed"),
    )


@_storage_errors
def plan_remove(selector: ModelSelector, options: StorageOptions | None = None, *, sctx=None) -> RemovalPlan:
    from sparkrun.core.model_usage import usage_blockers

    sctx = _context(sctx)
    selectors, recipe = _selectors(selector, sctx, removal=True)
    locations, _, _ = _locations(options or StorageOptions(), sctx, recipe)
    targets, blockers, seen = [], [], set()
    for location in locations:
        try:
            snapshot = _query(location, "preview", selectors=selectors)
        except WorkerError as error:
            message = "%s: %s" % (location.host, error)
            blockers.append(message)
            targets.append(
                RemovalTarget(
                    location,
                    selectors,
                    {
                        "selected": [
                            {
                                "repo_id": s["repo_id"],
                                "commits": [],
                                "action": "blocked",
                                "state": "blocked",
                                "cache_state": "unverified",
                                "allocated_bytes": None,
                                "reclaimable_bytes": None,
                                "error": str(error),
                            }
                            for s in selectors
                        ],
                        "blockers": [message],
                        "reclaimable_bytes": None,
                    },
                )
            )
            continue
        if snapshot["identity"] not in seen:
            seen.add(snapshot["identity"])
            selected = [s for s in snapshot["selected"] if s.get("action") != "already-absent"]
            if selected:
                snapshot["usage"] = _managed_usage(location, snapshot, sctx, selected)
                blockers.extend("%s: %s" % (location.host, b) for b in usage_blockers(snapshot["usage"]))
            targets.append(RemovalTarget(location, selectors, snapshot))
            blockers.extend("%s: %s" % (location.host, b) for b in snapshot["blockers"])
    return RemovalPlan(tuple(targets), tuple(blockers))


@_storage_errors
def remove(plan: RemovalPlan, *, sctx=None) -> OperationResult:
    """Preflight every cache under exclusion before deleting from any cache."""
    if plan.blockers:
        raise ModelInUse("; ".join(plan.blockers))
    from sparkrun.models.download import model_cache_path
    from sparkrun.core.model_usage import usage_blockers

    operation = uuid.uuid4().hex
    results = []
    with ExitStack() as stack:
        locked = []
        # Non-blocking locks also avoid deadlock when aliases order differently.
        for target in sorted(plan.targets, key=lambda t: t.snapshot["identity"]):
            # Ignore caller-provided path fields; derive canonical basenames.
            selectors = tuple(
                {
                    "repo_id": s["repo_id"],
                    "revision": s.get("revision"),
                    "cache_name": Path(model_cache_path(s["repo_id"], "/__cache_home__")).name,
                }
                for s in target.selectors
            )
            try:
                worker = stack.enter_context(WorkerSession(target.location.host, target.location.hf_home, target.location.ssh_kwargs))
                current = worker.request("lock", selectors=selectors, expected=target.snapshot)
                locked.append((target, worker, selectors, current))
            except WorkerError as error:
                raise _translate(error) from error
        context = _context(sctx)
        for target, _, _, current in locked:
            selected = [s for s in current["selected"] if s.get("action") != "already-absent"]
            if selected:
                blockers = usage_blockers(_managed_usage(target.location, current, context, selected))
                if blockers:
                    raise ModelInUse("%s: %s" % (target.location.host, "; ".join(blockers)))
        for target, worker, selectors, _ in locked:
            try:
                outcome = worker.request(
                    "remove",
                    selectors=selectors,
                    expected=target.snapshot,
                    operation=operation,
                    usage_scope="managed",
                    managed_checked=True,
                )
            except WorkerError as error:
                # Once submitted, an SSH failure cannot prove whether deletion
                # happened. Keep earlier successes and explicitly expose doubt.
                outcome = {"state": "indeterminate" if error.code == "StorageUnreachable" else "blocked", "error": str(error)}
            results.append({"location": target.location.to_dict(), **outcome})
    return OperationResult(tuple(results))


@_storage_errors
def plan_pull(selector: ModelSelector, options: StorageOptions | None = None, *, sctx=None) -> PullPlan:
    from sparkrun.core.hosts import is_control_in_cluster
    from sparkrun.models.revision import resolve_hub_commit
    from sparkrun.orchestration.ssh import should_run_locally

    sctx = _context(sctx)
    options = options or StorageOptions()
    inputs, recipe = _selectors(selector, sctx, pull=True)
    # --include-local is inventory/removal scope, not a distribution target.
    locations, cluster, local_cache = _locations(replace(options, include_local=False), sctx, recipe)
    prefs = cluster.distribution.model
    enabled = prefs.enabled and (recipe is None or recipe.distribution_config.models.enabled)
    local_head = should_run_locally(locations[0].host, locations[0].ssh_kwargs.get("ssh_user"))
    if len(locations) == 1 and local_head:
        local_cache = locations[0].hf_home
    mode = options.transfer_mode
    if mode not in {"auto", "local", "push", "delegated", "pull"}:
        raise InvalidModelSelection("Unknown model transfer mode")
    if mode == "auto":
        from sparkrun.orchestration.distribution import _is_cross_user

        mode = (
            "local"
            if is_control_in_cluster([loc.host for loc in locations]) and not _is_cross_user(locations[0].ssh_kwargs)
            else "delegated"
        )
    resolved, pins = [], {}
    for item in inputs:
        targets = item.get("target") or list(range(len(locations)))
        if targets == [-1]:
            targets = list(range(len(locations)))
        if any(type(i) is not int or i < 0 or i >= len(locations) for i in targets):
            raise InvalidModelSelection("Distribution target index out of range for the selected hosts")
        resolved.append({**item, "target": targets})
    # Validate all indices before any Hub call. A commit is immutable and needs
    # no online lookup; mutable refs are resolved once across all destinations.
    for item in resolved:
        if item["repo_id"] and enabled:
            key = (item["repo_id"], item["revision"])
            try:
                if key not in pins:
                    pins[key] = resolve_hub_commit(*key, config=sctx.config)
            except Exception as error:
                raise SparkrunError(str(error)) from error
            item["revision"] = pins[key]
    return PullPlan(
        locations,
        tuple(resolved),
        mode,
        str(local_cache),
        recipe_ref=str(selector.recipe) if isinstance(selector.recipe, str) else getattr(recipe, "name", None),
        preserve_perms=prefs.preserve_perms,
        skip_fan_out=prefs.skip_fan_out,
        enabled=enabled,
    )


@_storage_errors
def pull(plan: PullPlan, *, sctx=None) -> OperationResult:
    from sparkrun.core.cluster_manager import ModelDistributionPrefs
    from sparkrun.core.config import resolve_hf_token
    from sparkrun.core.model_preparation import prepare_model, quiet_model_preparation
    from sparkrun.core.model_storage_locks import protect_caches
    from sparkrun.models.download import download_model
    from sparkrun.orchestration.ssh import should_run_locally

    sctx = resolve_sctx(sctx)
    if not plan.enabled:
        return OperationResult(({"state": "disabled", "reason": "Model distribution explicitly disabled"},))
    if all(item.get("state") == "externally-managed" for item in plan.inputs):
        return OperationResult(tuple({"model": item["model"], "state": "externally-managed"} for item in plan.inputs))
    locations = [(loc.host, loc.hf_home, loc.ssh_kwargs) for loc in plan.locations]
    if plan.transfer_mode in {"local", "push"}:
        local = ("localhost", plan.local_cache_dir, {})
        if local not in locations:
            locations.append(local)
    for host, _, ssh in locations:
        try:
            setup_helper(host, ssh)
        except WorkerError as error:
            raise _translate(error) from error
    results = []
    repos = [{"repo_id": i["repo_id"], "revision": i["revision"]} for i in plan.inputs if i["repo_id"]]
    with quiet_model_preparation(), protect_caches(locations, repos, kind="pull", recipe_ref=plan.recipe_ref):
        for item in plan.inputs:
            if item.get("state") == "externally-managed":
                results.append({"model": item["model"], "state": "externally-managed"})
                continue
            selected = [plan.locations[index] for index in item["target"]]
            try:
                if len(selected) == 1 and should_run_locally(selected[0].host, selected[0].ssh_kwargs.get("ssh_user")):
                    rc = download_model(item["model"], cache_dir=selected[0].hf_home, revision=item["revision"], token=resolve_hf_token())
                    if rc:
                        raise SparkrunError("Model download failed (exit %s)" % rc)
                    failures = []
                else:
                    failures = prepare_model(
                        item["model"],
                        [loc.host for loc in selected],
                        full_hosts=[loc.host for loc in plan.locations],
                        cache_dir=selected[0].hf_home,
                        local_cache_dir=plan.local_cache_dir,
                        transfer_mode=plan.transfer_mode,
                        ssh_kwargs=selected[0].ssh_kwargs,
                        revision=item["revision"],
                        hf_token=resolve_hf_token(),
                        prefs=ModelDistributionPrefs(preserve_perms=plan.preserve_perms, skip_fan_out=plan.skip_fan_out),
                    )
                by_host = {failure.host: failure.reason for failure in failures}
                for loc in selected:
                    results.append(
                        {
                            "location": loc.to_dict(),
                            "model": item["model"],
                            "revision": item["revision"],
                            "state": "failed" if loc.host in by_host else "pulled",
                            "error": by_host.get(loc.host),
                        }
                    )
            except Exception as error:
                results.append({"model": item["model"], "state": "failed", "error": str(error)})
    return OperationResult(tuple(results))


@_storage_errors
def setup_storage(options: StorageOptions | None = None, *, offline=False, sctx=None) -> OperationResult:
    sctx = _context(sctx)
    locations, _, _ = _locations(options or StorageOptions(), sctx)
    results = []
    for location in locations:
        try:
            setup_helper(location.host, location.ssh_kwargs, offline=offline)
            results.append({"location": location.to_dict(), "state": "ready"})
        except WorkerError as error:
            results.append({"location": location.to_dict(), "state": "failed", "error": str(error)})
    return OperationResult(tuple(results))
