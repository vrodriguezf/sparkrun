"""Console-free model storage request/response types. JSON schema version 1."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Serializable:
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ModelSelector(Serializable):
    model: str | None = None
    recipe: Any = None
    revision: str | None = None
    include_dependencies: bool = False


@dataclass(frozen=True)
class StorageOptions(Serializable):
    cluster: Any = None
    hosts: tuple[str, ...] | None = None
    hosts_file: str | None = None
    local: bool = False
    include_local: bool = False
    cache_dir: str | None = None
    local_cache_dir: str | None = None
    transfer_mode: str = "auto"


@dataclass(frozen=True)
class CacheLocation(Serializable):
    host: str
    hf_home: str
    source: str
    ssh_kwargs: dict = field(default_factory=dict, repr=False)
    cluster: Any = field(default=None, repr=False, compare=False)

    @property
    def hub(self):
        return self.hf_home.rstrip("/") + "/hub"

    def to_dict(self):
        # Connection keys/options are internal, not part of reports.
        return {"host": self.host, "hf_home": self.hf_home, "hub": self.hub, "source": self.source}


@dataclass(frozen=True)
class InventoryReport(Serializable):
    caches: tuple[dict, ...]
    errors: tuple[dict, ...] = ()
    schema_version: int = 1

    @property
    def complete(self):
        return not self.errors

    def to_dict(self):
        return {**asdict(self), "complete": self.complete, "usage_scope": "managed"}


@dataclass(frozen=True)
class ModelReport(InventoryReport):
    selectors: tuple[dict, ...] = ()
    externally_managed: tuple[str, ...] = ()


@dataclass(frozen=True)
class RemovalTarget:
    location: CacheLocation
    selectors: tuple[dict, ...]
    snapshot: dict


@dataclass(frozen=True)
class RemovalPlan(Serializable):
    targets: tuple[RemovalTarget, ...]
    blockers: tuple[str, ...] = ()
    schema_version: int = 1

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "blockers": self.blockers,
            "targets": [{"location": t.location.to_dict(), "selectors": t.selectors, **t.snapshot} for t in self.targets],
            "scope": "Entire model cache directories, or only explicitly selected revisions; all GGUF variants in the selected scope",
            "usage_scope": "managed",
        }


@dataclass(frozen=True)
class OperationResult(Serializable):
    results: tuple[dict, ...]
    schema_version: int = 1

    @property
    def complete(self):
        return all(r["state"] in {"removed", "already-absent", "pulled", "externally-managed", "disabled", "ready"} for r in self.results)

    def to_dict(self):
        return {**asdict(self), "complete": self.complete}


RemovalResult = OperationResult
PullResult = OperationResult


@dataclass(frozen=True)
class PullPlan(Serializable):
    locations: tuple[CacheLocation, ...]
    inputs: tuple[dict, ...]
    transfer_mode: str
    local_cache_dir: str
    recipe_ref: str | None = None
    preserve_perms: bool = True
    skip_fan_out: bool = False
    enabled: bool = True
    schema_version: int = 1

    def to_dict(self):
        data = asdict(self)
        data["locations"] = [loc.to_dict() for loc in self.locations]
        data["staging_cache"] = self.local_cache_dir if self.transfer_mode in {"local", "push"} else None
        return data
