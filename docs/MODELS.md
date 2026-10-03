# Model storage

Model weights have a separate lifecycle from recipes, serving workloads,
container images, and runtime compilation caches. Removing a recipe, stopping a
job, or unloading a proxy endpoint does **not** delete its weights.

```bash
sparkrun model list --cluster mylab
sparkrun model list --cluster mylab --include-local --revisions --sort size
sparkrun model show Qwen/Qwen3-1.7B --cluster mylab
sparkrun model show --recipe my-recipe --cluster mylab

# Optional prefetch; run/proxy load still download missing weights.
sparkrun model pull --recipe my-recipe --cluster mylab
sparkrun model pull Qwen/Qwen3-1.7B --revision COMMIT --cluster mylab --dry-run

# Stop/unload serving first, then preview and remove retained files.
sparkrun model remove Qwen/Qwen3-1.7B --cluster mylab --dry-run
sparkrun model remove Qwen/Qwen3-1.7B --cluster mylab
```

The commands are `list`, `show`, `pull`, and `remove`, without aliases. The existing top-level
`sparkrun list` continues to list **recipes**, not downloaded models. LiteLLM
manages endpoints; it is not the authority for model storage.

Human-readable inventory uses one table per host/cache, with model ID,
attributed size, measured disk usage, revision count, cache state, managed usage and origin. `--revisions` adds
full commit hashes and refs; `--sort size` orders larger measured disk allocations first within
each table. Usage uncertainty and warnings remain outside the tables.
`--json` retains the structured report, without table formatting.

## Locations and scope

Commands accept `--cluster`, `--hosts/-H`, and `--hosts-file`. With none,
SparkRun uses the default cluster/hosts, falling back to the control machine
when none are configured. `--local` explicitly ignores defaults and cannot be
combined with remote selectors. `--include-local` includes a distinct control
copy for inspection/removal. Verified local aliases are deduplicated; matching
path strings on different machines are not proof of shared storage.

`--cache-dir` means the **HF home**, not the `hub` directory. For example,
`--cache-dir /data/huggingface` inspects `/data/huggingface/hub`.
`--local-cache-dir` selects a separate control copy/staging cache. Reports show
resolved hosts and Hub paths. Recipe cache overrides retain launch precedence
and trust checks. The head node's resolved cache path is used on workers,
matching the existing launch/distribution path.

Inventory scans the filesystem: downloads remain visible even when the recipe,
registry or job record is gone. Preparation receipts add provenance, not proof
of ownership. Old downloads without receipts have `origin unknown`. Current
path resolution is used; if a cache location changed, pass its old HF home
explicitly to inspect/remove that copy.

Absolute model paths and `cluster_config.resolved_model_path` are externally
managed and never removed. Provider-managed volumes require a storage adapter;
this release never substitutes the controller's cache for a provider volume.

## Removal

A bare repository ID selects **the entire model cache directory** on each host:
all cached revisions, all GGUF variants, refs, local metadata, orphan blobs and
partial downloads. This works for recognized caches and damaged/incomplete
caches whose paths can be safely inspected. Broken metadata alone does not
prevent removing a clearly identified model directory. Recipes, container images,
other model directories and SparkRun's coordination records remain untouched.

`--revision` selects a locally cached ref or commit without contacting the Hub.
Different hosts can resolve a ref to different commits; the preview shows each
host's selection. A revision selector removes only those snapshots, their refs
and unshared blobs, **even for the last recognized revision**; orphan files and
unrelated metadata remain. Invalid metadata or partial downloads block revision
removal with guidance to inspect the cache or explicitly select the whole model.
It never silently widens to whole-model removal.

The preview uses a per-model table with one row per host/cache, including blocked
and unreachable targets. It reports cache condition, observed contents, disk
usage, estimated reclamation, intended action and managed usage. Details include
the exact repository directory, snapshot directory IDs, GGUF paths/variants,
refs, partial files and problems such as a ref pointing to a missing snapshot.
`model show` lists observed snapshot filenames even when HF rejects the metadata.
**Present snapshot directories/files are not proof of complete or usable weights.**
For example, a broken `main` ref can coexist with hundreds of GiB of actual weights.

`--recipe` selects the primary model. A pinned recipe selects its cached
revision; an unpinned recipe selects every cached revision of its primary repo.
`--include-dependencies` opts into also selecting draft/other model inputs, each
with its own pin. Recipe names and repository IDs are not guessed interchangeably.

Selective GGUF file removal is not supported. `repo:quant` and quant-specific
recipes are rejected by `remove`; explicitly choose the bare repository only when
deleting **every** variant is intended.

`--dry-run` performs real, read-only probes: no helper installation, cache
directory creation, pending-record pruning, completion snapshots or deletion.
Normal removal prints the plan and asks once, naming the number of target caches
and hosts. Scripts and JSON-mode removal
require `--yes`; it skips only confirmation, never safety checks.

All targets are preflighted before deletion starts. SparkRun acquires cache-local
advisory exclusion, rechecks usage and the cache fingerprint, reconstructs the
deletion operation from selectors, validates every path, executes it, and verifies
the resulting filesystem. Whole-model deletion verifies the exact directory is
gone; revision deletion verifies the selected paths are gone. A changed cache requires a fresh preview.
An absent repo is an idempotent success; an unknown revision of a present repo
is a selection error.

Multi-host deletion is not transactional. A lost SSH reply after submission can
leave an `indeterminate` result alongside successful hosts. Inspect again and
retry; never infer success from a lost connection. Target-side journals retain
attempted/completed selections.

### Safety boundaries

- The tested backend is `huggingface-hub==1.8.0`, the existing project pin.
  Repository-local blobs/snapshots/refs are supported. Cross-repository shared
  blob stores, escaping/directory symlinks, symlinked repository roots, nested
  mounts and special files are **blocked**. Whole-model removal allows partial
  downloads and dangling internal file links, with the same usage/exclusion
  checks as recognized caches. Its directory is derived from the validated model
  ID, never an arbitrary deletion path from the caller. Revision removal requires
  valid metadata and uses the pinned HF strategy without its whole-repo shortcut.
  Newer shared stores require a tested backend upgrade.
- Removal supports recognized local Linux filesystems. NFS/CIFS and other
  unverified shared filesystems are blocked: observing every consumer and
  establishing cross-client exclusion require an additional backend/coordinator.
- Usage checks have an explicit **managed-only** scope: all inference and
  preparation must go through participating SparkRun clients. They query the
  live Docker/native executor status used by cluster and proxy operations,
  not arbitrary processes' memory maps. Protected SSH/PAM sessions therefore
  do not block deletion. External containers, manual inference and external
  downloads are outside this scope; coordinate them separately.
- Usage is reported per repository: `in-use`, `preparing`,
  `no-managed-consumers`, or `unknown`. An active model does not block deleting
  another model with verified non-use. Launch bindings under
  `hub/.locks/sparkrun/bindings/` record model dependencies (including drafts),
  cache identity, recipe, job and live workload instances. Docker IDs and
  native boot/PID/start-time identities prevent reuse of stale bindings.
  A saved binding alone is not evidence that a job is still running; proxy
  registration and request traffic are not liveness tests either.
  Recorded custom native PID directories are probed as well as the selected
  cluster's configured executor namespaces. Pre-existing native jobs in other,
  unregistered directories must be stopped before using this managed scope;
  arbitrary hidden state directories cannot be discovered automatically.
- An explicitly pinned, observed commit allows revision-level matching.
  Mutable or unpinned inputs protect the whole repository; recorded cached
  revisions are not a claim that a particular revision was loaded. Inputs are
  protected conservatively on every serving host, even when distribution
  targeted a subset. Custom runtimes must declare dependencies through the
  model-input hook; arbitrary extra downloads are not inferred from commands.
- Missing backend coverage, unreachable hosts, changed workload instances and
  running jobs without verified bindings remain `unknown` and block removal.
  Existing live jobs must be stopped or relaunched with this version; they are
  not silently adopted from current recipes or old controller metadata. Legacy
  pending records remain conservative blockers. An active job bound only in a
  different cache also remains unknown here rather than guessing its paths.
- Updated run, proxy/benchmark launches through the common launcher, recipe
  strategy preparation, build staging and prefetch publish durable target-side
  reservations. They exclude deletion on cache hits and throughout the
  download-to-activation interval. Competing preparations of one repository
  are rejected; nested phases share a reference-counted reservation. Receipts
  record observed cached revisions and survive serving shutdown. Successful
  launches publish instance bindings before releasing preparation guards.
  Binding failures and failed launch results retain their reservations.
  Removal acquires every selected cache guard and repeats live usage checks
  before any deletion, excluding concurrent managed starts.
- Reservations surviving failed/interrupted controllers never expire by age or
  PID alone. An operator must investigate the reported operation and confirm
  its controller, transfers and workloads have stopped before recovering its
  exact record under `hub/.locks/sparkrun/reservations/`. Automatic stale-record
  repair and force deletion are not provided. Do not remove the coordination
  directory wholesale.
- Older clients, external Hugging Face tools, arbitrary custom processes and
  unconfigured shared-cache consumers do not honor SparkRun's protocol. These
  checks are not a global guarantee against arbitrary external writers.
  Coordinate those clients before deleting weights.

**Attributed** sizes can count data used by several revisions and can be unknown
when metadata is damaged. **Disk usage** is independently measured using allocated
filesystem blocks, counting each inode once within a repository, including orphan
and partial files. Whole-model reclaim estimates cover that entire tree; revision
estimates cover only the selected paths. File inodes with hardlinks outside the
deletion set are excluded from the reclamation estimate. These estimates are not
an audit of filesystem free-space changes: open files, snapshots/reflinks and
other activity can change actual savings.
Free space is measured on the cache filesystem, not assumed to be `/`.

## Prefetch

`model pull` is weights-only: no GPU scheduling, image pulls/builds, runtime
preparation hooks, tuning, page-cache clearing, serving, or proxy registration.
Dependencies use a pure runtime hook shared with launches. Each entry's pin and
target indices are preserved; indices refer to the ordered selected host set.
This can stage more copies than a later occupancy-based launch selects.

Existing transfer implementations handle `local`, `push`, `delegated` and
per-node `pull`, including permission and skip-fan-out preferences. Automatic
prefetch chooses local staging when the controller is a selected node, otherwise
head/delegated download. Unlike launch auto-mode, prefetch does not probe
InfiniBand just to select a mode; choose an explicit mode for control staging.

Mutable Hub refs are resolved once per repository/revision so destinations get
the same commit. Explicit commits need no metadata lookup. `--dry-run` can query
Hub metadata but never transfers weights. Local/push operations retain a control
staging copy; `--include-local` governs inventory/removal, not source requirements.

## Helper setup

No global `hf` executable is needed. The same versioned Python worker runs
locally and over SSH without installing all of SparkRun on nodes. It uses the
tested Hugging Face dependency, preferring a compatible existing environment.
Preparations provision it as needed. Prepare an existing cluster explicitly
before its first inventory:

```bash
sparkrun setup model-storage --cluster mylab
sparkrun setup model-storage --cluster mylab --offline
```

The private helper lives at
`~/.cache/sparkrun/model-storage/hf-1.8.0/` on each target. Offline setup requires
an existing `uv` and cached wheels (or configuration such as `UV_FIND_LINKS`).
List/show/removal previews never bootstrap it; unavailable/incompatible helpers
produce actionable errors.

## Python and JSON

```python
from sparkrun.api import model

options = model.StorageOptions(local=True)
inventory = model.list_cached(options)
plan = model.plan_remove(model.ModelSelector(model="org/model"), options)
# Show plan.to_dict() and obtain frontend confirmation before model.remove(plan).
```

Also available: `inspect`, `plan_pull`, `pull`, `remove`, and `setup_storage`.
The API emits no terminal UI; confirmation is separate from backend checks.
Failures derive from `SparkrunError`; execution results preserve per-location
failures. JSON has `schema_version: 1`, numeric byte fields, coverage/uncertainty
and structured errors. Removal rows expose `action` (`remove-repository`,
`remove-revisions`, `already-absent`, or `blocked`), `cache_state`, `path`,
`allocated_bytes`, `reclaimable_bytes`, observed `contents` and diagnostic `issues`.
An empty recognized-commit list never means a damaged directory is empty or skips
its usage checks. Reports exclude SSH keys/options, HF tokens, full recipe
environments and provider credentials.

Exit codes: `0` successful/absent, `1` blocked/incomplete/partial, `2` invalid
selection or missing noninteractive confirmation, `130` interrupted.
