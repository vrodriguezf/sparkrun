"""``sparkrun.api.status`` — inspect running workloads via the resolved Executor.

The status surface routes through the *same* executor resolution chain
the launcher uses (CLI > recipe > cluster > runtime > SparkrunConfig),
so the inspector matches the launcher: whatever would have run a
workload is what's asked about it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sparkrun.core.cluster_manager import ClusterDefinition, ClusterStatusResult
    from sparkrun.core.cluster_status import ClusterStatus
    from sparkrun.core.context import SparkrunContext

logger = logging.getLogger(__name__)


def status(
    hosts: list[str],
    *,
    executor: str | None = None,
    cluster: "str | ClusterDefinition | None" = None,
    ssh_kwargs: dict | None = None,
    sctx: "SparkrunContext | None" = None,
    read_only: bool = False,
) -> "ClusterStatus":
    """Return a :class:`ClusterStatus` snapshot of *hosts*.

    Args:
        hosts: Host list to inspect.
        executor: Optional executor name override (CLI-level).  When
            ``None``, the executor is resolved via the standard chain
            with *cluster* providing the cluster row.
        cluster: Named cluster or pre-loaded definition.  When set,
            the cluster's ``executor`` / ``executor_config`` and
            ``hosts_hardware`` flow into the resolution and the
            query.  When ``None``, default executor is used and
            hardware falls back to DGX Spark per host.
        ssh_kwargs: Per-key overrides for cluster-scoped SSH configuration.
            Explicit None/empty values clear that configured key.
        sctx: Optional shared :class:`SparkrunContext`.  Provides
            cluster manager + SAF variables for chained-call sharing.
        read_only: Do not persist the shell-completion running snapshot.

    Returns:
        A :class:`ClusterStatus` snapshot.  Unreachable hosts are
        omitted from :attr:`ClusterStatus.hosts`; callers can detect
        this with ``status.for_host(h) is None``. Use ``observation_errors``
        to detect incomplete peer-backend coverage on otherwise reachable hosts.
    """
    from sparkrun.api._resolve import scope_operation, resolve_cluster
    from sparkrun.orchestration.executor import query_status_for_cluster

    # Always end up with a populated ClusterDefinition; hosts are the
    # explicit list passed in.
    cluster_def = resolve_cluster(cluster, hosts, sctx=sctx)
    # Refresh provider-backed connection details before any SSH (no-op for ssh).
    sctx, ssh_kwargs = scope_operation(cluster_def, sctx=sctx, ssh_kwargs=ssh_kwargs)
    v = sctx.variables
    config = sctx.config
    host_hardware = cluster_def.hosts_hardware or None

    # The single status source: query every enabled executor on this cluster's
    # substrate (its ``status_scope``) and merge.  For an SSH cluster that's
    # docker + local (disjoint state on the same hosts); for a provider cluster
    # (modal / k8s) it's that provider alone.  See ``query_status_for_cluster``.
    snapshot = query_status_for_cluster(
        cluster_def,
        list(hosts),
        executor=executor,
        ssh_kwargs=ssh_kwargs,
        host_hardware=host_hardware,
        config=config,
        v=v,
    )
    if not read_only:
        _record_running_snapshot(snapshot, sctx)
    return snapshot


def _record_running_snapshot(snapshot: "ClusterStatus", sctx) -> None:
    """Leave the sweep's answer behind for shell completion to read.

    Completion cannot sweep for itself — it runs on every TAB, and a host that
    no longer resolves would hang the terminal.  Recording it here, at the one
    place every occupancy sweep passes through, means completion gets a live
    view without ever opening a connection.

    The observation retains successful coverage per executor/destination.
    A successful peer backend cannot erase another backend's missing coverage.
    """
    try:
        from sparkrun.orchestration.job_metadata import save_running_snapshot

        save_running_snapshot(snapshot.observation, sctx=sctx)
    except Exception:
        logger.debug("Could not record running snapshot", exc_info=True)


def status_report(
    hosts: list[str],
    *,
    executor: str | None = None,
    cluster: "str | ClusterDefinition | None" = None,
    ssh_kwargs: dict | None = None,
    cache_dir: str | None = None,
    sctx: "SparkrunContext | None" = None,
) -> "ClusterStatusResult":
    """Return a display-oriented :class:`ClusterStatusResult` for *hosts*.

    The higher tier over :func:`status`: it takes the occupancy
    :class:`~sparkrun.core.cluster_status.ClusterStatus` snapshot ``status``
    produces and classifies it into cluster groups vs solo entries, enriches
    each with cached job metadata, and derives idle hosts + relevant pending
    ops.  Programmatic occupancy consumers (schedulers, discovery) should call
    :func:`status` for the lean snapshot; the CLI display paths (``cluster
    status``, ``stop --all``) call this.

    Args mirror :func:`status`, plus:
        cache_dir: Cache directory for job metadata + pending ops.  Falls back
            to the resolved context's configured cache directory.

    Returns:
        A :class:`ClusterStatusResult`.
    """
    from sparkrun.api._context import resolve_sctx
    from sparkrun.core.cluster_manager import classify_cluster_status

    sctx = resolve_sctx(sctx)
    snapshot = status(hosts, executor=executor, cluster=cluster, ssh_kwargs=ssh_kwargs, sctx=sctx)

    from sparkrun.core.config import resolve_configured_cache_dir

    cache_dir = str(resolve_configured_cache_dir(cache_dir, config=sctx.config))

    return classify_cluster_status(snapshot, cache_dir=cache_dir, host_list=list(hosts))


__all__ = ["status", "status_report"]
