"""Model usage derived from live managed workloads, not arbitrary OS processes.

Bindings are target-side provenance, never evidence of liveness on their own.
Only matching live executor instance identities can make a binding authoritative.
"""

from __future__ import annotations

COVERAGE = "SparkRun-managed workloads and preparations only; external processes/downloads are not covered"


def workload_instances(workload):
    """Stable backend identities: Docker IDs or native boot/PID/start ticks."""
    if not workload.containers or any(not c.instance_id or not c.executor for c in workload.containers):
        return None
    return sorted({c.executor + ":" + c.instance_id for c in workload.containers})


def reservation_matches(record, repo_id):
    repos = record.get("repos")
    # Legacy, empty or malformed reservations cannot establish narrower scope.
    return (
        not isinstance(repos, list)
        or not repos
        or any(not isinstance(repo, dict) or not repo.get("repo_id") or repo["repo_id"] == repo_id for repo in repos)
    )


def managed_usage(snapshot, host, cache, selections, pending=()):
    """Return cache/per-repository usage for the selected cached commits.

    Missing backend coverage or legacy/unidentifiable live jobs stays unknown.
    Historical bindings and idle proxy registrations do not reserve weights.
    """
    unknown = list(snapshot.observation_errors.values())
    entry = snapshot.for_host(host)
    if entry is None:
        unknown.append("Incomplete managed workload observation on " + host)
    claims = []
    reserved_operations = {r.get("operation") for r in cache.get("usage", {}).get("reservations", []) if r.get("operation")}
    pending = [p for p in pending if p.get("storage_operation") not in reserved_operations]
    for workload in entry.workloads if entry else ():
        instances = workload_instances(workload)
        matches = [b for b in cache.get("bindings", []) if b.get("job") == workload.cluster_id]
        binding = matches[0] if len(matches) == 1 else None
        if (
            not instances
            or not binding
            or binding.get("instances") != instances
            or binding.get("identity") != cache["identity"]
            or not isinstance(binding.get("repos"), list)
        ):
            unknown.append("Active job %s has no verified model binding; stop or relaunch it with this version" % workload.cluster_id)
            continue
        # A broken record is uncertainty, not proof that the job uses nothing.
        repos = binding["repos"]
        if not repos or any(
            not isinstance(r, dict)
            or not isinstance(r.get("repo_id"), str)
            or not r["repo_id"]
            or (
                r.get("commits") is not None
                and (
                    not isinstance(r["commits"], list)
                    or not r["commits"]
                    or any(not isinstance(c, str) or len(c) != 40 for c in r["commits"])
                )
            )
            for r in repos
        ):
            unknown.append("Invalid model binding for active job " + workload.cluster_id)
            continue
        claims.append(binding)

    models = []
    for selected in selections:
        repo_id, commits = selected["repo_id"], set(selected.get("commits", []))
        jobs = [
            {"job": b["job"], "recipe": b.get("recipe"), "runtime": b.get("runtime")}
            for b in claims
            if any(
                r["repo_id"] == repo_id
                and (
                    selected.get("action") in {"remove-repository", "blocked"}
                    or r.get("commits") is None
                    or commits.intersection(r["commits"])
                )
                for r in b["repos"]
            )
        ]
        reservations = [r for r in cache.get("usage", {}).get("reservations", []) if reservation_matches(r, repo_id)]
        uncertainty = list(unknown)
        # Legacy central markers do not enumerate draft/dependency models. Do
        # not infer that a different primary model means this repo is unused.
        uncertainty.extend("Legacy pending operation %s needs reconciliation" % p.get("operation", "unknown") for p in pending)
        state = "preparing" if reservations else "in-use" if jobs else "unknown" if uncertainty else "no-managed-consumers"
        models.append({"repo_id": repo_id, "state": state, "jobs": jobs, "reservations": reservations, "uncertainty": uncertainty})
    states = {m["state"] for m in models}
    state = next((s for s in ("preparing", "in-use", "unknown") if s in states), "unknown" if unknown else "no-managed-consumers")
    return {
        "scope": "managed",
        "coverage": COVERAGE,
        "state": state,
        "models": models,
        "uncertainty": sorted(set(unknown + [u for m in models for u in m["uncertainty"]])),
        "reservations": cache.get("usage", {}).get("reservations", []),
    }


def usage_blockers(usage):
    blockers = []
    for model in usage["models"]:
        repo = model["repo_id"]
        for job in model["jobs"]:
            blockers.append(
                "%s is in use by job %s (recipe %s); stop or proxy unload first" % (repo, job["job"], job["recipe"] or "unknown")
            )
        for reservation in model["reservations"]:
            blockers.append("%s is preparing in operation %s" % (repo, reservation.get("operation", "unknown")))
        blockers.extend("%s: %s" % (repo, u) for u in model["uncertainty"])
    return tuple(dict.fromkeys(blockers))
