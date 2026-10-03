"""Cluster status dataclasses — the shape of "what's running where?"

This module is **data only**.  Production of :class:`ClusterStatus`
lives behind the :class:`~sparkrun.orchestration.executors._base.Executor`
ABC's ``query_status`` method — there is no separate extension point
for status providers.  Each Executor knows how to inspect its own
backend (Docker via ``docker ps``, K8s via ``kubectl get pods``, …).

Consumers (the ``occupancy-sparse`` and ``occupancy-dense`` schedulers,
the ``sparkrun status`` CLI, the ``cluster monitor`` TUI) all flow
through ``sparkrun.api.status`` → ``executor.query_status``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from sparkrun.core.status_observation import ExecutorCoverage, RunningSnapshot
from sparkrun.core.allocations import GpuAllocation


@dataclass(frozen=True)
class ContainerDetail:
    """Per-container / per-process realization of a workload on one host.

    The concrete backend object behind a :class:`RunningWorkload` — a
    Docker container (``docker ps``) or a native process (LocalExecutor
    pidfile).  Carries the presentation-layer detail the ``sparkrun
    status`` CLI needs that the occupancy-shaped :class:`RunningWorkload`
    discards: the container ``name``, its ``role`` (``solo`` / ``head`` /
    ``worker`` / ``node_<rank>``) derived from the name, a human ``status``
    string (e.g. ``"Up 10 seconds"``), and the ``image``.
    """

    name: str
    role: str
    status: str
    image: str
    executor: str = ""
    """Registered name of the executor that observed this container/process.

    Stamped by ``query_status_for_cluster`` on each executor's snapshot before
    the merge, so it survives the docker+local fold.  This is what lets
    teardown reach the *right* substrate: ``stop --all`` discovers workloads
    across every executor sharing a status scope, and a container reported by
    ``docker ps`` must be removed with ``docker rm`` while a native process
    reported from a pidfile must be signalled.  Empty means unattributed
    (a snapshot built directly in a test, or a pre-existing serialization) —
    callers fall back to the cluster's default executor.
    """

    allocations: tuple[GpuAllocation, ...] = ()
    """Worker-persisted reservations; empty means legacy/unknown, never shared."""

    instance_id: str = ""
    """Backend instance identity, not a reusable workload name (storage safety)."""


@dataclass(frozen=True)
class RunningWorkload:
    """A single sparkrun-launched workload occupying slots on one host.

    :attr:`memory_used_gb` and :attr:`util_fraction` are optional
    per-workload resource accounting fields populated by status
    providers with per-workload allocation information. They describe reserved
    capacity, not current GPU activity. Worker-persisted allocations in
    ``containers`` take precedence when present; legacy unknown claims never
    establish permission to share.
    """

    cluster_id: str
    intent_id: str | None = None
    """Hex intent identifier (the deterministic prefix of cluster_id).

    Recovered from the ``sparkrun.intent_id`` container label when
    emitted; otherwise derived from the cluster_id's intent prefix or
    enriched from cached job metadata.  ``None`` indicates a workload
    whose container name does not parse as a canonical sparkrun
    identifier.
    """
    recipe_name: str | None = None
    runtime_name: str | None = None
    started_at: float | None = None
    ranks_on_host: int = 1
    container_ids: tuple[str, ...] = field(default_factory=tuple)
    memory_used_gb: float | None = None
    util_fraction: float | None = None
    containers: tuple[ContainerDetail, ...] = field(default_factory=tuple)
    """Per-container/process realizations of this workload on the host.

    Optional (default empty for back-compat).  Populated by executors
    that can observe per-container detail (docker ``Names``/``Status``/
    ``Image``, local pidfiles); consumed by the ``sparkrun status`` CLI.
    """


@dataclass(frozen=True)
class TerminationInfo:
    """What became of a workload that :class:`ClusterStatus` no longer reports.

    The post-mortem peer of :class:`RunningWorkload`.  ``query_status`` answers
    "what is running"; this answers, for something that is *not*, whether its
    remains are still on the substrate and how the operator inspects them.

    Produced by
    :meth:`~sparkrun.orchestration.executors._base.Executor.describe_terminated`,
    because every part of the answer is substrate-specific: a stopped Docker
    container, a `local` pidfile whose process is gone, a Failed k8s Pod.
    """

    exists: bool | None
    """Whether the workload's remains are still present on the substrate.

    - ``True`` — still there (a stopped container, a leftover pidfile, a Failed
      Pod), so there is something to inspect.
    - ``False`` — **confirmed** gone.
    - ``None`` — could not be determined.

    Callers must treat ``None`` as inconclusive and never as ``False``: the
    difference decides whether cached job metadata is deleted, and an
    unreachable host must not be able to trigger that.
    """

    detail: str | None = None
    """Substrate-native state, e.g. ``"Exited (137) 3 minutes ago"``."""

    investigate_hints: tuple[str, ...] = field(default_factory=tuple)
    """Substrate-native commands the operator can run next.

    The executor owns the wording because it owns the substrate — ``docker
    logs`` is wrong advice on a k8s cluster and meaningless for a ``local``
    job.  Callers render these; they never author them.
    """


@dataclass(frozen=True)
class GpuOccupancy:
    """Per-accelerator occupancy detail on a host.

    Populated by status providers that can introspect per-GPU usage
    (e.g. via ``nvidia-smi --query-gpu=memory.used,utilization.gpu``).
    Consumed by fractional-capable schedulers to decide whether a new
    rank's :class:`~sparkrun.core.scheduler.ResourceRequest` fits on
    each accelerator.
    """

    gpu_index: int
    used_memory_gb: float = 0.0
    used_util_fraction: float = 0.0
    workloads: tuple[RunningWorkload, ...] = field(default_factory=tuple)

    exclusive: bool = False
    """An exclusive allocation owns this GPU even when the engine is idle."""


@dataclass(frozen=True)
class HostOccupancy:
    """Per-host accelerator occupancy at the time of the snapshot.

    :attr:`gpus` carries per-accelerator detail when the producing
    executor can supply it; otherwise it's empty and consumers should
    fall back to host-level :attr:`used_slots` / :attr:`free_slots`.
    """

    host: str
    workloads: tuple[RunningWorkload, ...] = field(default_factory=tuple)
    used_slots: int = 0
    free_slots: int = 0
    free_memory_gb: float | None = None
    gpus: tuple[GpuOccupancy, ...] = field(default_factory=tuple)

    @property
    def total_slots(self) -> int:
        """Total accelerator slots = used + free."""
        return self.used_slots + self.free_slots


@dataclass(frozen=True)
class ClusterStatus:
    """Snapshot of cluster occupancy produced by an Executor.

    ``hosts`` carries one entry per host inspected (in input order).
    ``executor`` is the registered name of the executor that produced
    the snapshot (e.g. ``"docker"``) — primarily useful for diagnostics
    and asymmetry detection (e.g. asking the docker executor about a
    k8s-managed host).
    """

    hosts: tuple[HostOccupancy, ...] = field(default_factory=tuple)
    queried_at: float = 0.0
    executor: str = ""
    errors: dict[str, str] = field(default_factory=dict)
    """Hosts the executor could not reach, mapped to an error message.

    Populated by executors when a host's remote probe fails (non-zero
    return code / no result) — those hosts are absent from :attr:`hosts`,
    so this is how a caller distinguishes an *unreachable* host from a
    reachable-but-idle one.  Merged as a union across executors, minus any
    host that turns out reachable via at least one of them (see
    :meth:`merged_with`). Target-resolution failures are also retained. Use
    :attr:`observation_errors` for the complete set of incomplete observations.
    """

    coverage: tuple[ExecutorCoverage, ...] = ()
    """Successful coverage per executor; absent for unscoped/legacy snapshots."""

    @property
    def observation(self) -> RunningSnapshot:
        return RunningSnapshot(frozenset(self.running_cluster_ids()), self.coverage)

    @property
    def observation_errors(self) -> dict[str, str]:
        """Incomplete observations, including failed peers on a reachable host."""
        errors = dict(self.errors)
        for coverage in self.coverage:
            for host in coverage.requested_hosts - coverage.hosts:
                errors.setdefault(host, "%s status was not observed" % coverage.target.executor)
        return errors

    def for_host(self, host: str) -> HostOccupancy | None:
        """Return the :class:`HostOccupancy` for *host*, or ``None`` if absent."""
        for entry in self.hosts:
            if entry.host == host:
                return entry
        return None

    def free_slots(self, host: str) -> int:
        """Free accelerator slots on *host*; 0 when host is absent."""
        entry = self.for_host(host) if host not in self.observation_errors else None
        return entry.free_slots if entry is not None else 0

    def running_cluster_ids(self) -> tuple[str, ...]:
        """Distinct cluster_ids running anywhere in the snapshot."""
        seen: list[str] = []
        for entry in self.hosts:
            for w in entry.workloads:
                if w.cluster_id not in seen:
                    seen.append(w.cluster_id)
        return tuple(seen)

    @classmethod
    def merge(cls, snapshots: "list[ClusterStatus]") -> "ClusterStatus":
        """Fold *snapshots* (in priority order) into one via :meth:`merged_with`.

        The fold is left-associative: ``snapshots[0]`` is authoritative and
        wins any per-``cluster_id`` collision (callers pass the cluster's
        default executor first).  Used slots sum across executors; capacity is
        the shared host hardware; containers union.  Returns an empty snapshot
        when *snapshots* is empty and the sole snapshot unchanged (same object)
        when there's exactly one, so a single-executor scope is zero-cost.
        """
        if not snapshots:
            return cls()
        result = snapshots[0]
        for snap in snapshots[1:]:
            result = result.merged_with(snap)
        return result

    def merged_with(self, other: "ClusterStatus") -> "ClusterStatus":
        """Merge *other* into this snapshot, preserving this snapshot's host order.

        Combines two snapshots of the (same) host set produced by different
        executors (e.g. ``docker`` + ``local``, which inspect *disjoint*
        backend state on the same hosts) into one honest view.  Per host
        present in both:

        - ``workloads`` are concatenated, deduped by ``cluster_id`` (this
          snapshot's entry wins on collision);
        - ``used_slots`` is this snapshot's used plus the peer's slots for
          only its *non-duplicate* workloads — a ``cluster_id`` present in
          both (e.g. the same recipe launched once via docker and once via
          local yields the same deterministic id) is counted once, matching
          the workload dedup above (``used_slots == sum(ranks_on_host)`` for
          both producing executors);
        - ``free_slots`` recomputes as ``max(capacity - used_slots, 0)`` where
          ``capacity`` is this snapshot's ``total_slots`` (both snapshots share
          host hardware, so capacity matches).

        A host present in only one snapshot is carried over unchanged.  This
        snapshot's ``queried_at`` and ``executor`` name are kept (the merged
        snapshot reports as the primary executor to keep single-name consumers
        working).
        """
        other_by_host = {entry.host: entry for entry in other.hosts}
        seen_hosts: set[str] = set()
        merged: list[HostOccupancy] = []
        for entry in self.hosts:
            seen_hosts.add(entry.host)
            peer = other_by_host.get(entry.host)
            if peer is None:
                merged.append(entry)
                continue
            workloads = list(entry.workloads)
            index_by_id = {w.cluster_id: i for i, w in enumerate(workloads)}
            # Add the peer's slots only for workloads not already counted in
            # this snapshot, so a shared cluster_id isn't double-counted.  For a
            # shared cluster_id the primary workload is kept but its
            # ``containers`` absorb the peer's disjoint realizations (docker +
            # local), deduped by name — a shared workload can have both a Docker
            # container and a native process.
            peer_added_slots = 0
            for w in peer.workloads:
                idx = index_by_id.get(w.cluster_id)
                if idx is None:
                    index_by_id[w.cluster_id] = len(workloads)
                    workloads.append(w)
                    peer_added_slots += w.ranks_on_host
                elif w.containers:
                    existing = workloads[idx]
                    merged_containers = _union_containers(existing.containers, w.containers)
                    if merged_containers != existing.containers:
                        workloads[idx] = replace(existing, containers=merged_containers)
            used = entry.used_slots + peer_added_slots
            capacity = entry.total_slots  # shared host hardware → == peer.total_slots
            merged.append(
                with_gpu_allocations(
                    HostOccupancy(
                        host=entry.host,
                        workloads=tuple(workloads),
                        used_slots=used,
                        free_slots=max(capacity - used, 0),
                        free_memory_gb=entry.free_memory_gb,
                        gpus=_merge_gpu_observations(entry.gpus, peer.gpus),
                    ),
                    capacity,
                )
            )
        for entry in other.hosts:
            if entry.host not in seen_hosts:
                merged.append(entry)
        # Errors are the union of both snapshots (this snapshot wins on
        # collision), minus any host that is reachable via at least one
        # executor — a host present in the merged ``hosts`` is not an error.
        reachable = {entry.host for entry in merged}
        merged_errors: dict[str, str] = {}
        for host, msg in list(self.errors.items()) + list(other.errors.items()):
            if host in reachable or host in merged_errors:
                continue
            merged_errors[host] = msg
        return ClusterStatus(
            hosts=tuple(merged),
            queried_at=self.queried_at,
            executor=self.executor,
            errors=merged_errors,
            coverage=self.coverage + other.coverage,
        )


def attribute_executor(status: "ClusterStatus", executor_name: str) -> "ClusterStatus":
    """Return *status* with each container attributed to *executor_name*.

    Applied to every executor's snapshot before
    :meth:`ClusterStatus.merge` folds them, because after the fold a single
    workload can carry containers from two substrates (docker + local share
    the ``"host"`` scope) and only the producing executor can tear its own
    down.  Containers that already name an executor are left alone, so an
    executor that attributes its own output stays authoritative.

    Returns the same object when there is nothing to stamp.
    """
    if not executor_name:
        return status

    changed = False
    hosts: list[HostOccupancy] = []
    for entry in status.hosts:
        workloads: list[RunningWorkload] = []
        for w in entry.workloads:
            if any(not c.executor for c in w.containers):
                workloads.append(
                    replace(
                        w,
                        containers=tuple(c if c.executor else replace(c, executor=executor_name) for c in w.containers),
                    )
                )
                changed = True
            else:
                workloads.append(w)
        hosts.append(replace(entry, workloads=tuple(workloads)))

    if not changed:
        return status
    return replace(status, hosts=tuple(hosts))


def _union_containers(
    primary: tuple[ContainerDetail, ...],
    peer: tuple[ContainerDetail, ...],
) -> tuple[ContainerDetail, ...]:
    """Deduplicate the same realization, retaining distinct executor processes."""
    seen = {(c.name, c.executor) for c in primary}
    extra = [c for c in peer if (c.name, c.executor) not in seen]
    if not extra:
        return primary
    return primary + tuple(extra)


def workload_matches_intent(workload: RunningWorkload, intent_id: str) -> bool:
    """``True`` when *workload* belongs to the launch intent *intent_id*.

    The one answer to "is this workload mine?", shared by the two callers that
    ask it for opposite reasons: placement subtracts its own intent's
    workloads from the occupancy snapshot (so a relaunch doesn't treat its own
    containers as foreign load), while ``--ensure`` looks for exactly those
    workloads to decide there is nothing to launch.  They must agree, or
    ``--ensure`` would decline to launch something placement had already
    decided to replace.

    Matches on :attr:`RunningWorkload.intent_id` when the executor could
    recover it (container label / cached job metadata), else on the intent
    prefix parsed out of the ``cluster_id``.  A workload whose name is not a
    canonical sparkrun identifier matches nothing — deliberately: an
    unidentifiable container is not evidence that *this* intent is running.
    """
    if not intent_id:
        return False
    if workload.intent_id == intent_id:
        return True
    try:
        from sparkrun.orchestration.job_metadata import parse_cluster_id

        return parse_cluster_id(workload.cluster_id)[0] == intent_id
    except Exception:
        return False


def empty_status(hosts: list[str], executor: str = "") -> ClusterStatus:
    """Build a zero-occupancy snapshot — every host fully free, no workloads.

    Used as a safe default by executors that don't implement
    introspection (and as a sentinel in tests).
    """
    return ClusterStatus(
        hosts=tuple(HostOccupancy(host=h) for h in hosts),
        queried_at=0.0,
        executor=executor,
    )


def with_gpu_allocations(occupancy: HostOccupancy, capacity: int) -> HostOccupancy:
    """Combine worker reservations with existing observations without freeing unknown slots."""
    if not any(c.allocations for w in occupancy.workloads for c in w.containers):
        return occupancy
    gpu_workloads: dict[int, list[RunningWorkload]] = {}
    reservations: dict[int, list[GpuAllocation]] = {}
    unknown = max(0, occupancy.used_slots - sum(w.ranks_on_host for w in occupancy.workloads))
    for workload in occupancy.workloads:
        records = {}
        conflicting = False
        for container in workload.containers:
            for allocation in container.allocations:
                key = (container.executor, allocation.rank)
                if key in records and records[key] != allocation:
                    conflicting = True
                records[key] = allocation
        # Missing/conflicting records cannot establish free capacity. Distinct
        # executor realizations reserve separately, even for the same rank/job.
        if (
            not workload.containers
            or conflicting
            or any(not c.allocations for c in workload.containers)
            or any(a.gpu_index >= capacity for a in records.values())
        ):
            unknown += max(1, workload.ranks_on_host)
            continue
        for a in records.values():
            reservations.setdefault(a.gpu_index, []).append(a)
            if workload not in gpu_workloads.setdefault(a.gpu_index, []):
                gpu_workloads[a.gpu_index].append(workload)
    observed = {g.gpu_index: g for g in occupancy.gpus}
    gpus = []
    for index in range(capacity):
        old = observed.get(index, GpuOccupancy(index))
        records = reservations.get(index, [])
        exclusive = bool(unknown) or old.exclusive or any(a.exclusive for a in records)
        reserved_util = 1.0 if exclusive else sum(a.util_fraction for a in records)
        reserved_mem = sum(a.memory_gb or 0 for a in records)
        workloads = tuple(dict.fromkeys(w.cluster_id for w in (*old.workloads, *gpu_workloads.get(index, []))))
        by_id = {w.cluster_id: w for w in (*old.workloads, *gpu_workloads.get(index, []))}
        gpus.append(
            GpuOccupancy(
                gpu_index=index,
                used_memory_gb=max(old.used_memory_gb, reserved_mem),
                used_util_fraction=max(old.used_util_fraction, reserved_util),
                workloads=tuple(by_id[cid] for cid in workloads),
                exclusive=exclusive,
            )
        )
    used = sum(bool(g.workloads or g.exclusive or g.used_util_fraction or g.used_memory_gb) for g in gpus)
    return replace(occupancy, gpus=tuple(gpus), used_slots=used, free_slots=max(0, capacity - used))


def _merge_gpu_observations(primary, secondary):
    result = {g.gpu_index: g for g in primary}
    for gpu in secondary:
        old = result.get(gpu.gpu_index)
        if old is None:
            result[gpu.gpu_index] = gpu
            continue
        old_ids = {w.cluster_id for w in old.workloads}
        overlapping = any(w.cluster_id in old_ids for w in gpu.workloads)
        combine = max if overlapping else lambda a, b: a + b
        result[gpu.gpu_index] = replace(
            old,
            workloads=old.workloads + tuple(w for w in gpu.workloads if w.cluster_id not in old_ids),
            used_memory_gb=combine(old.used_memory_gb, gpu.used_memory_gb),
            used_util_fraction=combine(old.used_util_fraction, gpu.used_util_fraction),
            exclusive=old.exclusive or gpu.exclusive,
        )
    return tuple(result[index] for index in sorted(result))
