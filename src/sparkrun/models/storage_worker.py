"""Versioned, standalone cache worker (stdlib + exactly huggingface-hub 1.8.0).

This file is shipped verbatim over SSH stdin; do not import SparkRun here.
Never accepts deletion paths from the caller. A removal session reconstructs
whole-repository or HF revision operations from selectors, validates their
scope, and verifies the result.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import stat
import sys
import time
import uuid

PROTOCOL = 1
BACKEND = "1.8.0"
MAX_REQUEST = 1024 * 1024


class StorageError(Exception):
    def __init__(self, message, code="StorageCapabilityMissing"):
        super().__init__(message)
        self.code = code


def hub_path(home):
    path = Path(home).expanduser()
    if not path.is_absolute() or ".." in path.parts or path == Path(path.anchor):
        raise StorageError("Expected an absolute HF home, not a root or relative path", "UnsafeStoragePath")
    hub = path / "hub"
    # Do not follow cache-root aliases: a changed symlink would change the lock.
    for parent in (hub, *hub.parents):
        if parent.is_symlink():
            raise StorageError("Symlinked cache roots are not supported: %s" % parent, "UnsafeStoragePath")
    return hub


def backend():
    try:
        import huggingface_hub
    except ImportError as exc:
        raise StorageError("Run sparkrun setup model-storage on this host (missing cache helper)") from exc
    if huggingface_hub.__version__ != BACKEND:
        raise StorageError("Cache helper requires huggingface-hub==%s; run sparkrun setup model-storage" % BACKEND)
    return huggingface_hub


def scan_hf(hub):
    hf = backend()
    try:
        return hf.scan_cache_dir(hub) if hub.exists() else hf.HFCacheInfo(size_on_disk=0, repos=frozenset(), warnings=[])
    except (OSError, ValueError, RuntimeError) as error:
        # Damaged metadata must not conceal the physical cache. Revision-only
        # removal still requires a successful HF scan; whole-repo removal does not.
        return hf.HFCacheInfo(size_on_disk=0, repos=frozenset(), warnings=[hf.CorruptedCacheException(str(error))])


def validate_selector(selector):
    repo = selector.get("repo_id", "")
    if not isinstance(repo, str) or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*(/[A-Za-z0-9_][A-Za-z0-9_.-]*)?", repo):
        raise StorageError("Invalid model repository ID", "InvalidModelSelection")
    if ".." in repo or "--" in repo or repo.endswith(".git"):
        raise StorageError("Invalid model repository ID", "InvalidModelSelection")
    revision = selector.get("revision")
    if revision is not None and (not isinstance(revision, str) or not revision or "\0" in revision or len(revision) > 1024):
        raise StorageError("Invalid revision", "InvalidModelSelection")
    return repo, revision


def identity(hub):
    info = hub.stat() if hub.exists() else None
    machine = Path("/etc/machine-id")
    machine_id = machine.read_text().strip() if machine.exists() else socket.gethostname()
    return "%s:%s:%s:%s" % (machine_id, os.getuid(), info.st_dev if info else "missing", info.st_ino if info else str(hub))


def filesystem(hub):
    """Only local filesystems have a supported exclusion/consumer boundary."""
    existing = hub
    while not existing.exists():
        existing = existing.parent
    fstype = "unknown"
    mounts = Path("/proc/self/mountinfo")
    if mounts.exists():
        best = -1
        for line in mounts.read_text().splitlines():
            left, right = line.split(" - ", 1)
            mount = Path(re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), left.split()[4]))
            if existing.is_relative_to(mount) and len(str(mount)) > best:
                best, fstype = len(str(mount)), right.split()[0]
    return {
        "type": fstype,
        "free_bytes": shutil.disk_usage(existing).free,
        "coordinated": fstype in {"ext4", "xfs", "btrfs", "zfs", "tmpfs", "overlay"},
    }


def tree_entries(repo):
    """Inspect names/inodes only, never following directory links or mounts."""
    if not repo.exists() and not repo.is_symlink():
        return []
    device = repo.parent.stat().st_dev
    # mountinfo catches bind mounts on the same device too (ismount does not).
    mounts = {
        Path(re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), line.split()[4]))
        for line in Path("/proc/self/mountinfo").read_text().splitlines()
    }
    entries = []
    paths = [repo]
    while paths:
        path = paths.pop()
        st = path.lstat()
        if st.st_dev != device or path in mounts:
            raise StorageError("Mounted path inside model cache: %s" % path, "UnsafeStoragePath")
        entries.append((path, st))
        if stat.S_ISDIR(st.st_mode):
            paths.extend(path.iterdir())
    return sorted(entries, key=lambda item: str(item[0]))


def tree_measurement(repo, entries):
    rows, allocated, inodes = [], 0, {}
    for path, st in entries:
        link = os.readlink(path) if stat.S_ISLNK(st.st_mode) else None
        rows.append(
            (str(path.relative_to(repo)), st.st_dev, st.st_ino, st.st_mode, st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_nlink, link)
        )
        key = (st.st_dev, st.st_ino)
        if key not in inodes:
            allocated += getattr(st, "st_blocks", 0) * 512
            inodes[key] = [st, 0]
        inodes[key][1] += 1
    # Exclude file inodes with links outside the deletion set from the estimate.
    reclaimable = sum(
        getattr(st, "st_blocks", 0) * 512 for st, links in inodes.values() if stat.S_ISDIR(st.st_mode) or links >= st.st_nlink
    )
    digest = hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest() if entries else "absent"
    return digest, allocated, reclaimable


def tree_state(repo):
    digest, allocated, _ = tree_measurement(repo, tree_entries(repo))
    return digest, allocated


def describe_tree(repo):
    """Filesystem evidence, independent of HF metadata validity/completeness."""
    entries = tree_entries(repo)
    digest, allocated, reclaimable = tree_measurement(repo, entries)
    snapshots, refs, partials, issues = {}, [], [], []
    files, blobs = 0, 0
    for path, st in entries:
        relative = path.relative_to(repo)
        parts = relative.parts
        if not parts:
            continue
        if parts[0] == "snapshots" and len(parts) >= 2:
            if len(parts) == 2 and stat.S_ISDIR(st.st_mode):
                snapshots.setdefault(parts[1], [])
            elif len(parts) > 2 and not stat.S_ISDIR(st.st_mode):
                snapshots.setdefault(parts[1], []).append(str(Path(*parts[2:])))
        if not stat.S_ISDIR(st.st_mode):
            files += 1
            blobs += int(parts[0] == "blobs")
            if path.name.endswith(".incomplete"):
                partials.append(str(relative))
        if parts[0] == "refs" and stat.S_ISREG(st.st_mode):
            # Never follow ref links or read arbitrarily large metadata.
            value = path.read_text(errors="replace").strip() if st.st_size <= 1024 else "<oversized ref>"
            refs.append({"name": str(Path(*parts[1:])), "commit": value})
    for ref in refs:
        if ref["commit"] not in snapshots:
            issues.append("Ref %s -> %s has no snapshot directory" % (ref["name"], ref["commit"]))
    if partials:
        issues.append("%d partial download file(s) present" % len(partials))
    return {
        "fingerprint": digest,
        "allocated_bytes": allocated,
        "reclaimable_bytes": reclaimable,
        "contents": {
            "files": files,
            "blobs": blobs,
            "snapshots": [{"commit": commit, "files": names} for commit, names in sorted(snapshots.items())],
            "refs": refs,
            "partial_files": partials,
        },
        "issues": issues,
    }


def validate_tree(repo, *, whole_repo=False):
    """Constrain every entry, including untracked files in whole-repo removal."""
    if repo.is_symlink():
        raise StorageError("Repository directory is a symlink: %s" % repo, "UnsafeStoragePath")
    if not repo.is_dir():
        raise StorageError("Model cache is not a directory: %s" % repo, "UnsafeStoragePath")
    for path, st in tree_entries(repo):
        if stat.S_ISLNK(st.st_mode):
            resolved = path.resolve(strict=not whole_repo)
            if not resolved.is_relative_to(repo) or resolved.is_dir() or (resolved.exists() and not resolved.is_file()):
                raise StorageError("Shared/external or directory symlink is not supported for removal: %s" % path, "UnsafeStoragePath")
        elif not stat.S_ISDIR(st.st_mode) and not stat.S_ISREG(st.st_mode):
            raise StorageError("Special file in model cache: %s" % path, "UnsafeStoragePath")
        if path.name.endswith(".incomplete") and not whole_repo:
            raise StorageError("Incomplete download needs investigation: %s" % path)


def read_records(hub, kind):
    parent = hub / ".locks" / "sparkrun" / kind
    for component in (hub / ".locks", parent.parent, parent):
        if component.is_symlink():
            raise StorageError("Symlinked coordination directory", "UnsafeStoragePath")
    if not parent.exists():
        return []
    if parent.is_symlink():
        raise StorageError("Symlinked coordination directory", "UnsafeStoragePath")
    records = []
    for path in sorted(parent.glob("*.json")):
        if path.is_symlink() or path.stat().st_size > MAX_REQUEST:
            raise StorageError("Invalid storage record: %s" % path)
        try:
            record = json.loads(path.read_text())
            if not isinstance(record, dict):
                raise ValueError("Storage record must be an object")
            records.append(record)
        except (ValueError, OSError) as exc:
            raise StorageError("Unreadable storage record: %s" % path) from exc
    return records


def write_record(hub, kind, key, value):
    parent = hub / ".locks" / "sparkrun" / kind
    for component in (hub / ".locks", parent.parent, parent):
        if component.is_symlink():
            raise StorageError("Symlinked coordination directory", "UnsafeStoragePath")
        component.mkdir(mode=0o700, exist_ok=True)
    key = hashlib.sha256(key.encode()).hexdigest()
    path = parent / (key + ".json")
    temporary = parent / (key + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x") as stream:
        os.chmod(temporary, 0o600)
        json.dump(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return path


def scan(hub):
    backend()
    result = {
        "hub": str(hub),
        "identity": identity(hub),
        "filesystem": filesystem(hub),
        "models": [],
        "warnings": [],
        "receipts": read_records(hub, "receipts"),
        "bindings": read_records(hub, "bindings"),
        "schema_version": PROTOCOL,
        "observed_at": time.time(),
    }
    if not hub.exists():
        return result
    if not hub.is_dir():
        raise StorageError("Hub cache is not a directory", "UnsafeStoragePath")
    cache = scan_hf(hub)
    result["warnings"] = [str(w) for w in cache.warnings]
    known = set()
    for repo in sorted(cache.repos, key=lambda r: r.repo_id):
        if repo.repo_type != "model":
            continue
        known.add(repo.repo_path.name)
        details = describe_tree(repo.repo_path)
        revisions = [
            {
                "commit": rev.commit_hash,
                "refs": sorted(rev.refs),
                "attributed_bytes": rev.size_on_disk,
                "files": sorted(str(f.file_path.relative_to(rev.snapshot_path)) for f in rev.files),
            }
            for rev in sorted(repo.revisions, key=lambda r: r.commit_hash)
        ]
        result["models"].append(
            {
                "repo_id": repo.repo_id,
                "path": str(repo.repo_path),
                "revisions": revisions,
                "attributed_bytes": repo.size_on_disk,
                **details,
                "state": "corrupt-or-incomplete" if details["issues"] else "cached",
            }
        )
    for path in sorted(hub.glob("models--*")):
        if path.name not in known:
            details = describe_tree(path)
            result["models"].append(
                {
                    "repo_id": path.name[8:].replace("--", "/"),
                    "path": str(path),
                    "revisions": [],
                    "attributed_bytes": None,
                    **details,
                    "state": "corrupt-or-incomplete",
                }
            )
    return result


def consumers(hub):
    """Target-side preparation state; live jobs are checked by the controller.

    Managed-only scope intentionally does not scan unrelated /proc processes or
    foreign containers. This is not evidence about arbitrary external clients.
    """
    reservations = read_records(hub, "reservations")
    return {
        "state": "preparing" if reservations else "awaiting-managed-observation",
        "scope": "managed",
        "uncertainty": [],
        "reservations": reservations,
        "coverage": "SparkRun-managed preparations; controller must verify live managed jobs",
    }


def selection(hub, selectors):
    backend()
    selected, strategies = [], []
    grouped = {}
    for selector in selectors:
        repo_id, revision = validate_selector(selector)
        group = grouped.setdefault(repo_id, {"cache_name": selector.get("cache_name", ""), "revisions": []})
        if group["cache_name"] != selector.get("cache_name", ""):
            raise StorageError("Conflicting cache basenames", "InvalidModelSelection")
        group["revisions"].append(revision)
    for repo_id, group in grouped.items():
        # Supplied by the controller's canonical model_cache_path implementation.
        name = group["cache_name"]
        if name != "models--" + repo_id.replace("/", "--"):
            raise StorageError("Invalid cache basename", "InvalidModelSelection")
        path = hub / name
        if not path.exists() and not path.is_symlink():
            selected.append(
                {
                    "repo_id": repo_id,
                    "path": str(path),
                    "commits": [],
                    "state": "already-absent",
                    "action": "already-absent",
                    "cache_state": "absent",
                    "fingerprint": "absent",
                    "allocated_bytes": 0,
                    "reclaimable_bytes": 0,
                }
            )
            continue
        whole_repo = None in group["revisions"]
        validate_tree(path, whole_repo=whole_repo)
        details = describe_tree(path)
        cache = scan_hf(hub)
        repo = next((r for r in cache.repos if r.repo_type == "model" and r.repo_id == repo_id), None)
        if repo is not None and repo.repo_path != path:
            raise StorageError("Repository/cache basename mismatch", "UnsafeStoragePath")
        base = {
            "repo_id": repo_id,
            "path": str(path),
            "state": "selected",
            **details,
            "cache_state": "cached" if repo is not None and not details["issues"] else "corrupt-or-incomplete",
        }
        if whole_repo:
            if not getattr(shutil.rmtree, "avoids_symlink_attacks", False):
                raise StorageError("Safe directory removal is not supported on this platform")
            selected.append(
                {**base, "action": "remove-repository", "commits": sorted(r.commit_hash for r in repo.revisions) if repo else []}
            )
            # Exact directory derived from the validated ID, never caller paths.
            strategies.append(path)
            continue
        if repo is None or details["issues"]:
            raise StorageError(
                "Cannot resolve a revision safely in this corrupt/incomplete cache; inspect it with model show, "
                "or explicitly remove the whole model without --revision: %s" % repo_id,
                "InvalidModelSelection",
            )
        commit_set = set()
        for revision in group["revisions"]:
            matches = {r.commit_hash for r in repo.revisions if revision is None or revision == r.commit_hash or revision in r.refs}
            if not matches:
                raise StorageError("Revision %r is not cached for %s" % (revision, repo_id), "InvalidModelSelection")
            commit_set.update(matches)
        commits = sorted(commit_set)
        # HF 1.8 matches hashes globally. Scope the cache first to avoid deleting
        # a different repository that happens to contain the same commit hash.
        assert cache is not None
        scoped = dataclasses.replace(cache, repos=frozenset([repo]))
        strategy = scoped.delete_revisions(*commits)
        if strategy.repos:
            # HF recursively deletes the whole repo for the last revision. A
            # revision selector must preserve orphan files/other local metadata.
            revisions = [r for r in repo.revisions if r.commit_hash in commit_set]
            strategy = dataclasses.replace(
                strategy,
                repos=frozenset(),
                snapshots=frozenset(r.snapshot_path for r in revisions),
                refs=frozenset(path / "refs" / ref for r in revisions for ref in r.refs),
                blobs=frozenset(f.blob_path for r in revisions for f in r.files),
            )
        for target in (*strategy.repos, *strategy.snapshots, *strategy.refs, *strategy.blobs):
            if not target.is_relative_to(path) or not target.resolve().is_relative_to(path):
                raise StorageError("HF strategy escaped the selected repository", "UnsafeStoragePath")
        entries = tree_entries(path)
        deletion_entries = [
            (p, st)
            for p, st in entries
            if p in strategy.blobs or p in strategy.refs or any(p.is_relative_to(snapshot) for snapshot in strategy.snapshots)
        ]
        _, _, reclaimable = tree_measurement(path, deletion_entries)
        selected.append({**base, "action": "remove-revisions", "commits": commits, "reclaimable_bytes": reclaimable})
        strategies.append(strategy)
    return selected, strategies


def preview(hub, selectors):
    usage = consumers(hub)
    all_reservations = read_records(hub, "reservations")
    repos = dict.fromkeys(s["repo_id"] for s in selectors)
    reservations = [
        r for r in all_reservations if not r.get("repos") or any(not p.get("repo_id") or p["repo_id"] in repos for p in r["repos"])
    ]
    usage = {**usage, "state": "preparing" if reservations else "awaiting-managed-observation", "reservations": all_reservations}
    blocked = []
    for record in reservations:
        blocked.append("Selected model is preparing in operation %s; finish preparation first" % record.get("operation", "unknown"))
    fs = filesystem(hub)
    if not fs["coordinated"]:
        blocked.append("Removal is not supported on filesystem %s (shared-cache coordination unverified)" % fs["type"])
    if (hub / "blobs").exists():
        blocked.append("Cross-repository shared blob store is not supported by the pinned removal backend")
    # Keep a visible row for every selected repo, even when inspection fails.
    selected = []
    for repo_id in repos:
        group = [s for s in selectors if s["repo_id"] == repo_id]
        try:
            rows, _ = selection(hub, group)
            selected.extend(rows)
        except (StorageError, OSError, ValueError, RuntimeError) as error:
            path = hub / ("models--" + validate_selector(group[0])[0].replace("/", "--"))
            try:
                details = describe_tree(path)
            except (OSError, StorageError, ValueError, RuntimeError):
                details = {"allocated_bytes": None}
            row = {
                **details,
                "repo_id": repo_id,
                "path": str(path),
                "commits": [],
                "state": "blocked",
                "action": "blocked",
                "cache_state": "unverified",
                "reclaimable_bytes": None,
                "error": str(error),
            }
            selected.append(row)
            blocked.append("%s: %s" % (repo_id, error))
    if all(item["action"] == "already-absent" for item in selected):
        blocked = []  # an absent repository requires no mutation or exclusion
    return {
        "schema_version": PROTOCOL,
        "hub": str(hub),
        "identity": identity(hub),
        "selected": selected,
        "usage": usage,
        "bindings": read_records(hub, "bindings"),
        "blockers": blocked,
        "reclaimable_bytes": sum(s["reclaimable_bytes"] for s in selected)
        if all(s["reclaimable_bytes"] is not None for s in selected)
        else None,
    }


def guard(hub):
    try:
        import fcntl
    except ImportError as exc:
        raise StorageError("Cache coordination requires POSIX advisory locks") from exc
    # Lock the directory inode, not a removable/recreatable lockfile. This also
    # permits a genuinely read-only preview and serializes legacy cache adoption.
    fd = os.open(hub, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError) as exc:
        os.close(fd)
        raise StorageError("Another storage operation holds this cache", "ModelInUse") from exc
    return fd


def remove_locked(hub, request):
    if request.get("usage_scope") != "managed" or request.get("managed_checked") is not True:
        raise StorageError("Removal requires a live managed-workload check under exclusion", "ModelInUse")
    current = preview(hub, request["selectors"])
    if current["blockers"]:
        raise StorageError("; ".join(current["blockers"]), "ModelInUse")
    expected = request["expected"]
    if current["identity"] != expected["identity"] or current["selected"] != expected["selected"]:
        raise StorageError("Cache changed after preview; create a new removal plan", "RemovalPlanChanged")
    _, strategies = selection(hub, request["selectors"])
    journal_id = request["operation"]
    write_record(hub, "journal", journal_id, {"state": "started", "selected": current["selected"], "at": time.time()})
    for strategy in strategies:
        if isinstance(strategy, Path):
            validate_tree(strategy, whole_repo=True)
            shutil.rmtree(strategy)
        else:
            strategy.execute()
    # execute() logs some failures without raising. Inspect the actual paths.
    for target in current["selected"]:
        path = Path(target["path"])
        remaining = [path] if target["action"] == "remove-repository" else [path / "snapshots" / c for c in target["commits"]]
        if any(p.exists() or p.is_symlink() for p in remaining):
            raise StorageError("Deletion verification failed for " + target["repo_id"], "StorageUnreachable")
    for strategy in strategies:
        if not isinstance(strategy, Path) and any(p.exists() or p.is_symlink() for p in (*strategy.blobs, *strategy.refs)):
            raise StorageError("Deletion verification failed for revision files", "StorageUnreachable")
    result = {"state": "removed" if strategies else "already-absent", "selected": current["selected"]}
    write_record(hub, "journal", journal_id, {**result, "at": time.time()})
    return result


def dispatch(request):
    if request.get("schema_version") != PROTOCOL:
        raise StorageError("Incompatible model storage protocol")
    hub = hub_path(request["hf_home"])
    action = request["action"]
    if action == "scan":
        return {**scan(hub), "usage": consumers(hub)}
    if action == "preview":
        return preview(hub, request["selectors"])
    if action == "reserve":
        hub.mkdir(parents=True, exist_ok=True)
        fd = guard(hub)
        try:
            existing = read_records(hub, "reservations")
            own = next((entry for entry in existing if entry["operation"] == request["operation"]), None)
            requested_repos = {entry["repo_id"] for entry in request["repos"]}
            for entry in existing:
                if entry["operation"] != request["operation"] and requested_repos.intersection(r["repo_id"] for r in entry["repos"]):
                    raise StorageError("Another preparation is using the selected repository", "ModelInUse")
            repos = list(request["repos"])
            if own:
                repos.extend(r for r in own["repos"] if r not in repos)
            write_record(
                hub,
                "reservations",
                request["operation"],
                {
                    "operation": request["operation"],
                    "repos": repos,
                    "at": time.time(),
                    "kind": request.get("kind", "preparation"),
                    "holders": own.get("holders", 1) + 1 if own else 1,
                },
            )
            return {"state": "reserved", "identity": identity(hub)}
        finally:
            os.close(fd)
    if action == "bind":
        fd = guard(hub)
        try:
            own = next((r for r in read_records(hub, "reservations") if r["operation"] == request["operation"]), None)
            if own is None:
                raise StorageError("Binding requires a live preparation reservation", "ModelInUse")
            requested = request["repos"]
            if not requested or any(validate_selector(r)[0] not in {p["repo_id"] for p in own["repos"]} for r in requested):
                raise StorageError("Binding is outside the reserved repositories", "InvalidModelSelection")
            observed = scan(hub)
            repos = []
            for item in requested:
                cached = next((m for m in observed["models"] if m["repo_id"] == item["repo_id"]), None)
                cached_commits = [r["commit"] for r in cached["revisions"]] if cached else []
                revision = item.get("revision")
                # A mutable/unpinned runtime may fetch a newer ref. Only an
                # explicit cached immutable pin can justify revision-level use.
                commits = [revision] if revision in cached_commits and re.fullmatch("[0-9a-f]{40}", revision or "") else None
                repos.append({"repo_id": item["repo_id"], "revision": revision, "commits": commits, "cached_revisions": cached_commits})
            binding = {
                "schema_version": 1,
                "job": request["job"],
                "recipe": request.get("recipe"),
                "runtime": request.get("runtime"),
                "instances": sorted(request["instances"]),
                "target": request.get("target", {"executor": "docker"}),
                "identity": identity(hub),
                "hub": str(hub),
                "repos": repos,
                "at": time.time(),
            }
            if not binding["instances"] or not all(isinstance(i, str) and i for i in binding["instances"]):
                raise StorageError("Missing live workload instances", "ModelInUse")
            write_record(hub, "bindings", request["job"], binding)
            return binding
        finally:
            os.close(fd)
    if action == "release":
        fd = guard(hub)
        try:
            records = read_records(hub, "reservations")  # validate metadata ancestors before unlink
            own = next((record for record in records if record["operation"] == request["operation"]), None)
            key = hashlib.sha256(request["operation"].encode()).hexdigest()
            path = hub / ".locks" / "sparkrun" / "reservations" / (key + ".json")
            if request.get("receipt"):
                receipt = request["receipt"]
                observed = scan(hub)
                receipt["cached_revisions"] = {
                    repo["repo_id"]: [rev["commit"] for rev in repo["revisions"]]
                    for repo in observed["models"]
                    if repo["repo_id"] in {r["repo_id"] for r in receipt.get("repos", [])}
                }
                write_record(hub, "receipts", request["operation"], receipt)
            if own and own.get("holders", 1) > 1:
                write_record(hub, "reservations", request["operation"], {**own, "holders": own["holders"] - 1})
            else:
                path.unlink(missing_ok=True)
            return {"state": "released"}
        finally:
            os.close(fd)
    raise StorageError("Unknown worker action")


def emit(value):
    print(json.dumps(value, separators=(",", ":")), flush=True)


def read_request():
    line = sys.stdin.readline(MAX_REQUEST + 1)
    if len(line) > MAX_REQUEST:
        raise StorageError("Storage request too large")
    return json.loads(line) if line else None


def main():
    held = None
    try:
        while (request := read_request()) is not None:
            try:
                if not isinstance(request, dict) or request.get("schema_version") != PROTOCOL:
                    raise StorageError("Incompatible model storage protocol")
                if request.get("action") == "lock":
                    if held is not None or request.get("schema_version") != PROTOCOL:
                        raise StorageError("Invalid lock session")
                    hub = hub_path(request["hf_home"])
                    # Missing caches have nothing to delete. They are rechecked
                    # at execution without creating anything during preflight.
                    held = (hub, guard(hub) if hub.exists() else None)
                    result = preview(hub, request["selectors"])
                    if result["blockers"]:
                        raise StorageError("; ".join(result["blockers"]), "ModelInUse")
                    if result["identity"] != request["expected"]["identity"] or result["selected"] != request["expected"]["selected"]:
                        raise StorageError("Cache changed after preview", "RemovalPlanChanged")
                elif request.get("action") == "remove":
                    if held is None or held[0] != hub_path(request["hf_home"]):
                        raise StorageError("Removal requires a locked session")
                    if held[1] is None:
                        if held[0].exists():
                            raise StorageError("Cache appeared after preview", "RemovalPlanChanged")
                        result = {"state": "already-absent"}
                    else:
                        result = remove_locked(held[0], request)
                else:
                    result = dispatch(request)
                emit({"ok": True, "result": result})
            except Exception as exc:
                emit({"ok": False, "error": {"code": getattr(exc, "code", "StorageUnreachable"), "message": str(exc)}})
    finally:
        if held is not None and held[1] is not None:
            os.close(held[1])


if __name__ == "__main__":
    main()
