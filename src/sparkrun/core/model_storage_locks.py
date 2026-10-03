"""Durable, cache-local launch/download reservations.

All reservation transitions and deletion use the same directory-inode flock.
Reservations intentionally do not expire: controller death or ambiguous SSH
failure is not evidence that its transfer or launch stopped. A reservation left
by a crashed client blocks deletion for operator investigation.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextlib import ExitStack
from contextvars import ContextVar
from functools import wraps
import logging
import time
import uuid

logger = logging.getLogger(__name__)
_operation_stack: ContextVar[ExitStack | None] = ContextVar("model_storage_operation", default=None)
_operation_id: ContextVar[str | None] = ContextVar("model_storage_id", default=None)
_operation_state: ContextVar[dict | None] = ContextVar("model_storage_state", default=None)


def guarded_operation(function):
    """Let a launch acquire protection after resolution and release on return."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        with ExitStack() as stack:
            token = _operation_stack.set(stack)
            operation_token = _operation_id.set(_operation_id.get() or uuid.uuid4().hex)
            state_token = _operation_state.set(_operation_state.get() or {"failed": False, "launch_reserved": False})
            try:
                result = function(*args, **kwargs)
                from sparkrun.core.launcher import LaunchResult

                state = _operation_state.get()
                if isinstance(result, LaunchResult) and state and state["launch_reserved"] and not kwargs.get("dry_run", False):
                    if result.rc:
                        state["failed"] = True
                    else:
                        record_launch_usage(result, sctx=kwargs.get("sctx"))
                return result
            finally:
                _operation_stack.reset(token)
                _operation_id.reset(operation_token)
                _operation_state.reset(state_token)

    return wrapped


def reserve_launch(recipe, runtime, hosts, cache_dir, local_cache_dir, ssh_kwargs, *, include_local=False, dry_run=False):
    """Protect cached launches too, not just paths that download missing files."""
    if dry_run:
        return
    from sparkrun.core.model_preparation import model_inputs
    from sparkrun.core.recipe import is_local_model_path

    inputs = model_inputs(recipe, runtime)
    repos = [
        {"repo_id": entry.name.split(":", 1)[0], "revision": entry.revision} for entry in inputs if not is_local_model_path(entry.name)
    ]
    if not repos:
        return
    locations = [(host, cache_dir, ssh_kwargs) for host in hosts]
    if include_local:
        locations.append(("localhost", local_cache_dir, {}))
    stack = _operation_stack.get()
    if stack is None:
        raise RuntimeError("Model preparation requires a guarded operation")
    stack.enter_context(protect_caches(locations, repos, kind="launch", recipe_ref=recipe.name))
    state = _operation_state.get()
    if state is not None:
        state["launch_reserved"] = True


def record_launch_usage(result, *, sctx=None):
    """Publish dependencies while preparation guards still exclude deletion.

    Store bindings on each target, so another controller can verify them. A
    binding failure retains guards; a stale job record never substitutes for a
    live executor observation. No proxy-specific liveness shortcut is used.
    """
    from sparkrun.api._status import status
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.model_preparation import model_inputs
    from sparkrun.core.model_usage import workload_instances
    from sparkrun.core.recipe import is_local_model_path
    from sparkrun.models.download import parse_gguf_model_spec
    from sparkrun.models.storage_remote import WorkerError, query
    from sparkrun.orchestration.primitives import build_ssh_kwargs

    hosts = list(result.host_list)
    ssh = build_ssh_kwargs(result.config)
    target = result.storage_target
    cluster = ClusterDefinition(name="", hosts=hosts, executor=target.get("executor"), executor_config=target.get("config", {}))
    snapshot = status(hosts, cluster=cluster, ssh_kwargs=ssh, sctx=sctx, read_only=True)
    if snapshot.observation_errors or any(snapshot.for_host(h) is None for h in hosts):
        raise WorkerError("Cannot verify launched model consumers; preparation reservations retained")
    inputs = model_inputs(result.recipe, result.runtime)
    repos = [{"repo_id": parse_gguf_model_spec(e.name)[0], "revision": e.revision} for e in inputs if not is_local_model_path(e.name)]
    for host in hosts:
        entry = snapshot.for_host(host)
        assert entry is not None
        live = next((w for w in entry.workloads if w.cluster_id == result.cluster_id), None)
        if live is None:
            if result.storage_detached:
                raise WorkerError("Detached job %s is not observable on %s; preparation reservations retained" % (result.cluster_id, host))
            continue  # a foreground workload may already have finished
        instances = workload_instances(live)
        if not instances:
            raise WorkerError("Cannot identify live job %s; preparation reservations retained" % result.cluster_id)
        query(
            host,
            result.effective_cache_dir,
            "bind",
            ssh_kwargs=ssh,
            operation=_operation_id.get(),
            job=result.cluster_id,
            recipe=result.recipe.qualified_name,
            runtime=result.runtime.runtime_name,
            repos=repos,
            instances=instances,
            target={
                "executor": target.get("executor", "docker"),
                **({"pid_dir": target["config"]["pid_dir"]} if target.get("executor") == "local" else {}),
            },
        )


def reserve_recipe_launch(recipe, runtime, hosts, options, cluster, sctx):
    """Cover recipe strategy hooks before the resolved common-launch phase."""
    if options.dry_run:
        return
    from sparkrun.orchestration.executor import cluster_status_scope
    from sparkrun.core.launcher import _enforce_recipe_mount_trust, resolve_recipe_trust, resolve_effective_cache_dir
    from sparkrun.orchestration.primitives import build_ssh_kwargs

    if cluster_status_scope(cluster, config=sctx.config, v=sctx.variables) != "host":
        return
    _enforce_recipe_mount_trust(recipe, resolve_recipe_trust(recipe, options.trust, sctx=sctx))
    cache = options.cache_dir or cluster.cache_dir
    local_cache = options.local_cache_dir or str(sctx.config.hf_cache_dir)
    if recipe.cluster_config:
        cache = recipe.cluster_config.remote_cache_dir or cache
        local_cache = recipe.cluster_config.local_cache_dir or local_cache
    ssh = build_ssh_kwargs(sctx.config)
    cache = resolve_effective_cache_dir(cache, hosts, ssh, sctx.config)
    reserve_launch(
        recipe,
        runtime,
        hosts,
        cache,
        local_cache,
        ssh,
        include_local=(options.transfer_mode or cluster.transfer_mode or "auto") in {"auto", "local", "push"},
    )


@contextmanager
def protect_caches(locations, repos, *, kind="preparation", recipe_ref=None):
    from sparkrun.models.storage_remote import query, setup_helper
    from sparkrun import __version__

    operation = _operation_id.get() or uuid.uuid4().hex
    operation_state = _operation_state.get()
    acquired = []
    complete = False
    started = False
    try:
        for host, home, ssh_kwargs in sorted(locations, key=lambda loc: (loc[0], loc[1])):
            setup_helper(host, ssh_kwargs)
            query(host, home, "reserve", ssh_kwargs=ssh_kwargs, operation=operation, repos=list(repos), kind=kind)
            acquired.append((host, home, ssh_kwargs))
        started = True
        yield
        complete = not (operation_state and operation_state["failed"])
    finally:
        # On an exception, retained reservations are deliberate. A failed SSH
        # launch may have started a live process; don't race a later removal.
        if not complete and started:
            if acquired:
                logger.warning(
                    "Storage reservation %s retained after incomplete %s; inspect affected caches before recovery", operation, kind
                )
        else:
            for host, home, ssh_kwargs in acquired:
                try:
                    query(
                        host,
                        home,
                        "release",
                        ssh_kwargs=ssh_kwargs,
                        operation=operation,
                        receipt={
                            "schema_version": 1,
                            "sparkrun_version": __version__,
                            "operation": operation,
                            "kind": kind,
                            "repos": list(repos),
                            "recipe": recipe_ref,
                            "hf_home": home,
                            "at": time.time(),
                        }
                        if complete
                        else None,
                    )
                except Exception:
                    logger.warning("Storage reservation %s retained on %s: release failed", operation, host, exc_info=True)
