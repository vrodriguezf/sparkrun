# Model storage management implementation plan

Status: initial implementation delivered on `feature/model-storage`, based on
`develop-next` (0.4.0). The original design below is retained as a checklist;
see [MODELS.md](MODELS.md) for the implemented interface and explicit limits.

2026-09-29 refinement: usage now has an explicit managed-only scope. Live
executor instances are matched to target-side per-model launch bindings;
unrelated OS processes no longer determine deletion safety. This supersedes
the generic consumer-scanning design below. Unknown/legacy live jobs remain
blocked; see MODELS.md for the current safety contract.

2026-10-02 refinement: bare model removal now explicitly removes the complete
validated repository directory, including damaged metadata, orphan blobs and
partial downloads. Per-host tables distinguish physical contents/disk allocation
from HF metadata validity, with blocked/unreachable targets kept visible.
Revision selectors remain narrow, including when selecting the final recognized
revision; they never fall back to whole-repository deletion. Empty commit lists
do not bypass usage checks. This supersedes the earlier blanket corrupt-cache
refusal; see MODELS.md for the current scope and safety boundaries.

Validation for this refinement: 10,856 passed / 52 skipped in the full suite,
plus 198 focused storage/documentation checks after final refinements. Ruff,
format checks and changed-module Pyright passed. The live two-node whole-model
dry-run reported both ~150.8 GiB copies and the dangling local ref; revision-only
preview kept the damaged node visible and blocked the plan. No deployed model
weights were deleted; destructive acceptance remains temporary-fixture-only.

Earlier validation: the full regression suite passed (10,829 passed, 52 skipped), plus
focused checks for final refinements. Lint/format checks and wheel packaging
passed. Destructive tests use temporary fixture caches, not deployed weights;
Live two-node inventory and a local Qwen removal dry-run passed with managed
usage reporting. Live inference/deletion acceptance and Python 3.13 validation
remain follow-up work.

Prepared: 2026-09-28. Repository baseline: `b444da8` (`main`, SparkRun 0.3.10).

## 1. Outcome and scope

Add a native `sparkrun model` command group for inspecting, downloading, and
removing model weights across a selected cluster. Users should not need to know
Hugging Face cache directory names, SSH into each node, or expose the `hf`
executable on their PATH.

The lifecycle becomes:

1. Discover or import a recipe.
2. Optionally prefetch its weights with `model pull`.
3. Launch with the existing `run` or `proxy load` command; these still fetch
   missing weights automatically.
4. Stop serving with the existing `stop` or `proxy unload` command.
5. Inspect retained downloads with `model list` and reclaim them with `model remove`.

The completed command group includes `list`, `show`, `pull`, and `remove`, backed by
one public Python API. Deliver inventory and safe removal first, then prefetching
through the same preparation path used by launches.

Initial execution support is local and SSH-accessible caches used by Docker and
native/local workloads, including the two-node Spark cluster. A provider whose
weights live in a managed volume needs an explicit storage backend; do not treat
its control machine's filesystem as that volume.

These commands manage model weights. Recipe definitions, container images,
runtime compilation caches, proxy aliases, and Hugging Face authentication files
remain separate resources. Arbitrary pre-placed directories are reported as
externally managed and are not deletion targets. Automatic eviction, bulk pruning,
and a graphical Storage page are follow-up work.

## 2. Findings that shape the implementation

| Existing code | Finding and consequence |
| --- | --- |
| `cli/__init__.py`, `cli/_recipe.py` | There is no model storage group or recipe install/uninstall command. Add a group without changing `sparkrun list`, which lists recipes. |
| `models/download.py` | Contains model ID/GGUF parsing, cache naming, revision lookup, and download helpers. Reuse their semantics. |
| `core/launcher.py` | Resolves effective cache locations, applies recipe path overrides, calls runtime preparation, and distributes weights. Storage operations must agree with these decisions. |
| `models/distribute.py`, `orchestration/distribution.py` | Support control-to-host, head-to-worker, and per-node downloads, plus shared-cache preferences. A download-only path should reuse these transfer primitives. |
| `core/recipe.py` | Distribution entries carry individual revisions and target node indices. A recipe can need several repositories. |
| `runtimes/vllm_distributed.py`, `runtimes/sglang.py` | Draft-model dependencies are currently added during `prepare()`. Discovery needs a pure dependency-resolution hook. |
| `orchestration/disk_info.py` | Provides aggregate cache sizes, but no per-model inventory. Free space is currently measured on `/`; new storage reporting must measure the cache's filesystem. |
| `api/_status.py`, `api/_jobs.py` | Live workload status and saved job metadata are available. Neither alone is a complete inventory of cached weights or their consumers. |
| `core/pending_ops.py` | Pending files are informational. A second caller can proceed while a marker exists, and markers expire. They are not sufficient exclusion for deletion. |
| `pyproject.toml` | Pins `huggingface-hub==1.8.0`. The installed library has `scan_cache_dir()` and revision deletion plans. Capabilities in newer online documentation cannot be assumed to exist in this pin. |
| `models/distribute.py` | Already recognizes caches with blobs shared across repositories and materializes external links when copying repositories. The removal backend must handle or explicitly reject these layouts. |
| `docs/PROXY.md`, `proxy/engine.py` | LiteLLM registration concerns serving endpoints. Storage management belongs in the shared SparkRun API and does not require a proxy or database. |

Paths in this table are relative to `src/sparkrun/`, except documentation and
`pyproject.toml`.

## 3. CLI contract

### Commands

```bash
# Inventory the selected cluster; include a separate control-machine cache.
sparkrun model list --cluster mylab
sparkrun model list --cluster mylab --include-local --json

# Inspect a downloaded repository or resolve a recipe to its model inputs.
sparkrun model show Qwen/Qwen3-1.7B --cluster mylab
sparkrun model show --recipe my-recipe --cluster mylab

# Prefetch weights without allocating GPUs or starting inference.
sparkrun model pull --recipe my-recipe --cluster mylab
sparkrun model pull Qwen/Qwen3-1.7B --revision COMMIT --cluster mylab

# Preview removal and then execute the same selection with confirmation.
sparkrun model remove Qwen/Qwen3-1.7B --cluster mylab --dry-run
sparkrun model remove Qwen/Qwen3-1.7B --cluster mylab
sparkrun model remove --recipe my-recipe --cluster mylab --dry-run

# Deliberately inspect or remove only the control-machine copy.
sparkrun model list --local
sparkrun model remove Qwen/Qwen3-1.7B --local
```

Use one name per command, without aliases. Keep the command group singular to
match `recipe`, `cluster`, and `registry`.

### Target and scope rules

- A positional target is a Hugging Face model repository ID, optionally using
  SparkRun's existing `repo:quant` syntax. Recipe lookup requires `--recipe`;
  do not guess whether a string names a recipe or repository.
- `MODEL` and `--recipe` are mutually exclusive. `--revision` applies to a direct
  model target. Recipe revisions come from resolved recipe/distribution entries.
- Reuse `--cluster`, `--hosts`, and `--hosts-file` resolution, SSH identity, and
  transport preparation. `--local` is mutually exclusive with remote selectors.
  Explicit host overrides retain the existing resolution semantics, including
  which cluster cache settings still apply.
- Without a selector, use configured default hosts/cluster, then local storage
  if none are configured. Every human-readable report starts with the resolved
  hosts and cache paths, including whether a default was used.
- `--include-local` adds a separate control-machine cache. It never silently
  expands deletion to every configured cluster. Deduplicate the control machine
  if it is already one of the targets.
- `--cache-dir` retains SparkRun's HF-home convention; show the corresponding
  resolved Hub directory explicitly. Reuse `--local-cache-dir` for a distinct
  control-machine cache. A future direct Hub-directory override must have an
  unambiguous name rather than overloading this convention.
- `--recipe` resolves the current recipe. Saved receipts show older revisions or
  repositories used by previous versions of that recipe, but do not silently add
  those old artifacts to the removal selection.
- Apply the existing recipe trust policy before honoring recipe-supplied host
  path/cache overrides. Reading an untrusted registry recipe must not authorize
  deletion in a different cache. Pure inspection must not run recipe hooks or
  mutate registry trust as a side effect.
- Shell completion reads only local completion data. It never connects to hosts
  or queries Hugging Face on a TAB press.

### `list` and `show`

The default inventory shows repository ID, cached bytes, locations, revision
count, and usage state. `show` adds exact paths, per-host details, cached refs,
quantization files, known recipe references, and active job IDs. Add `--revisions`
and `--sort size|name` to `list`; keep more elaborate filters out of the first
release.

Usage states are `in-use`, `preparing`, `no-managed-consumers`, and `unknown`.
Describe the coverage of the check; absence from the proxy is never evidence of
non-use. Recipe association is informational and must identify whether it comes
from a current recipe, saved receipt, or live workload.

Scan the filesystem even when there are no receipts. Existing downloads remain
visible after a recipe, registry, job record, or proxy entry is removed. Mark
unattributed downloads as `origin unknown`; do not claim SparkRun owns everything
in a shared Hugging Face cache.

A nonexistent cache is empty. An inaccessible cache is an error. Report corrupt
repositories and incomplete downloads as separate entries/warnings with whatever
size information can be obtained safely. Do not present a skipped corrupt repo
as zero disk usage or as automatically removable.

### `remove`

- Removing a bare repository ID selects all cached revisions of that repository
  in the chosen locations. The preview explicitly says this includes every
  cached GGUF variant in those revisions.
- `--revision` restricts the selection to a cached commit or locally resolved
  ref. No Hub connection is needed. If the same ref points to different commits
  on different hosts, show each commit separately.
- `--recipe` selects the primary model's cached pinned revision, or all cached
  revisions of its repository when it is unpinned. State this scope in the
  preview. Add `--include-dependencies` to opt into removing resolved draft and
  other distributed model inputs, each with its own revision policy.
- V1 removes repositories and whole revisions. A quant-specific target, including
  a recipe with `repo:quant`, must not silently become whole-repository deletion.
  If file-level deletion is not supported by the tested backend, reject it with
  an actionable message showing the bare repository command and its broader
  scope. Selective GGUF deletion is a subsequent capability, not a manual
  symlink/blob deletion workaround.
- Refuse removal when selected weights are in use or being prepared. Initially
  use repository-level usage protection even for revision-scoped deletion when
  the exact serving revision cannot be established. Suggest `stop` or
  `proxy unload`; do not stop workloads implicitly.
- `--dry-run` performs real read-only probes and prints the deletion plan. It
  neither deletes model files nor creates cache directories, repairs ownership,
  installs tooling, or prunes stale state. This differs from launch dry runs that
  can use synthetic paths; the help text must make that clear.
- Normal interactive execution prints the plan and asks once. `--yes` skips the
  prompt only; it does not bypass usage, identity, or path checks. Noninteractive
  execution requires `--yes` unless `--dry-run` is selected. There is no `--force`
  bypass in v1.
- If a repository is absent from an otherwise valid, reachable cache, report
  `already-absent`. Repeated removal is a successful no-op. An unresolvable
  revision in a repository that exists is an explicit selection error.
- A blocked target prevents any deletion during preflight. After execution has
  begun, an SSH failure can still produce a partial result; report it accurately
  and return nonzero. There is no promise of rollback across machines.

### `pull`

- Fetch the primary model and all declared downloadable dependencies. Honor
  individual revisions, GGUF selectors, per-entry host targets, transfer modes,
  permission preferences, and explicitly disabled distribution.
- A direct model ID is a weights-only operation and needs no runtime recipe.
- For recipe prefetching, target all explicitly selected hosts, independent of
  current GPU occupancy. Interpret explicit distribution target indices against
  that ordered host set. Explain that this can stage more copies than a future
  scheduled launch uses. Validate index bounds before transferring anything.
- Extract pure model dependency resolution and model preparation from the launch
  pipeline. Do not implement prefetch by invoking `api.run()` and stopping it,
  or by invoking a runtime's general `prepare()` hook.
- The command does not start containers, build/pull runtime images, execute recipe
  hooks, sync tuning, reserve GPUs, clear page caches, or alter proxy routes.
- A local transfer mode can leave a source copy on the control machine; show it
  in the plan/result and record it. `--include-local` controls inventory/removal,
  not whether a transfer is allowed to use a required staging cache.
- Resolve mutable remote refs to commit hashes once per repository in the pull
  plan so all new copies use the same revision. Cache inspection/removal remain
  offline; pulling may require Hub access and credentials.
- Use `--dry-run` to show destinations, dependencies, and transfer choices.
  Unknown remote download sizes remain unknown rather than becoming zero.
- For absolute-path/pre-placed inputs, report `externally-managed` and inspect
  existence where appropriate; never copy or delete them implicitly.

## 4. Shared API and module boundaries

Add `sparkrun.api.model`, separate from the existing `api/_models.py` dataclasses:

```text
api.model.list_cached(options, *, sctx=None) -> InventoryReport
api.model.inspect(selector, options, *, sctx=None) -> ModelReport
api.model.plan_pull(selector, options, *, sctx=None) -> PullPlan
api.model.pull(plan, *, sctx=None) -> PullResult
api.model.plan_remove(selector, options, *, sctx=None) -> RemovalPlan
api.model.remove(plan, *, sctx=None) -> RemovalResult
```

The public API is console-free, follows the existing `SparkrunContext` pattern,
raises typed `SparkrunError` subclasses for input/planning failures, and returns
structured per-target outcomes after execution begins. Confirmation belongs to
the CLI or future UI, while all safety checks belong to the API/backend.

| Proposed file or package | Responsibility |
| --- | --- |
| `api/model/__init__.py`, `_ops.py`, `_types.py` | Public operations, immutable plans/results, options, and serialization. |
| `models/storage.py` | Cache scan, selection, backend capability checks, deletion planning, and post-delete verification. |
| `models/storage_worker.py` | Standalone, versioned JSON request/response worker using only the standard library and the selected Hugging Face dependency; runnable locally or through SSH stdin. |
| `models/storage_remote.py` | Existing transport/SSH integration, helper discovery, bounded parallel probes, and worker-session lifecycle. |
| `core/model_preparation.py` | Pure dependency collection and shared model-only preparation receipt, analogous to `core/image_preparation.py`. |
| `core/model_storage.py` | Effective cache location resolution, cache identities, durable artifact receipts, and usage associations. |
| `core/model_storage_locks.py` | Cache coordination and launch reservations; separate from informational pending operations. |
| `cli/_model.py` | Thin Click wrappers, confirmation, and error/exit handling. |
| `utils/cli_formatters.py` | Inventory, detailed model view, removal preview, and per-host result formatting. |
| `cli/__init__.py`, `api/__init__.py` | Register/re-export the new command group and public API. |

Define at least these records:

- `CacheLocation`: host identity, SSH user, HF home, actual Hub root, resolution
  source, physical cache identity, filesystem information, helper capabilities.
- `CachedModel` / `CachedRevision`: repository ID, refs/commit, discovered files,
  per-location size, scan warnings, and timestamps when reliable.
- `ModelUsage`: job/operation identity, repository/revision knowledge, observation
  time, observation scope, recipe association, and uncertainty reason.
- `RemovalPlan`: normalized selectors, exact hosts/cache identities,
  repository/revision targets, estimated reclaimable bytes, warnings/blockers,
  and a fingerprint of the scanned state.
- `RemovalResult`: per-cache `removed`, `already-absent`, `blocked`, `failed`, or
  `indeterminate`; verified outcomes and measured/estimated byte fields.
- `PullPlan` / `PullResult`: resolved commit per dependency, source/destinations,
  transfer decisions, cache hits, and completed/failed destinations.

JSON output has `schema_version: 1`, integer byte fields, timestamps, structured
errors, and explicit completeness/coverage. Progress goes to stderr in JSON mode.
Never serialize tokens, entire recipe environments, or provider credentials.

Use typed failures such as `ModelInUse`, `StorageUnreachable`,
`StorageCapabilityMissing`, `UnsafeStoragePath`, and `RemovalPlanChanged`, all
under the existing `SparkrunError` hierarchy. Post-execution failures retain the
per-cache results rather than discarding completed work behind one exception.

A plan is a preview, not permission to delete arbitrary supplied paths. The
executor re-resolves identities and reconstructs the deletion strategy from
validated selectors before executing. Future HTTP clients must not be able to
submit a filesystem path list as a removal plan.

## 5. Cache locations, identity, and accounting

Extract the existing resolution rules into a reusable resolver and migrate
launch and storage callers together under parity tests. Preserve current
precedence: recipe `cluster_config` overrides, explicit launch options, applicable
cluster settings, then local configuration or the remote SSH user's environment.
Distinguish HF home from Hub cache root so `hub` is appended exactly once.

Important compatibility detail: the current remote launcher resolves the head
node's HF home and uses that effective path for distribution. Inventory must
include that actual destination on workers, even if their own environment would
suggest another path. Do not silently switch historical launches to different
per-node paths. Record resolved locations on new preparation receipts, surface
environment/path disagreements, and test direct `HF_HUB_CACHE` overrides against
actual downloader/container behavior before changing resolution semantics.

Use recorded locations for past operations and current resolved locations for
new ones. A recipe-specific override must be considered by `show --recipe` and
`remove --recipe`. Never substitute a control-machine path after a failed remote
probe.

Deduplicate locations by verified physical cache identity. Hostnames can be
aliases, the control machine can also be a worker, and two nodes can mount the
same NFS export under different paths. Matching path strings or observing that
both mounts are NFS does not establish shared identity.

Create cache identity/coordination metadata as part of explicit setup or a
mutating preparation operation, in a reserved location such as
`<hub>/.locks/sparkrun/`. Use a random cache ID plus verified filesystem/export
identity; a copied marker alone must not merge two independent caches. Read-only
inventory does not create markers. When sharing is uncertain, show separate
observations with an unverified total and block deletion until identity and
coordination can be established. Do not duplicate the same lock acquisition for
multiple aliases of one physical cache.

Report these quantities distinctly:

- Bytes attributed to a repository or revision, which may include shared blobs.
- Physical/cache total where it can be measured without double counting.
- Estimated bytes reclaimable by the specific removal plan.
- Free space on the filesystem containing the cache (`statvfs` or `df` against
  that path, not `/`).

Do not sum per-revision or per-repository sizes to promise disk savings. Account
for symlinks, hardlinks, and shared blobs using backend knowledge and filesystem
identity. Sparse files, snapshots, reflinks, and other filesystem activity mean
that an observed change in free space is not an exact audit of bytes reclaimed.

## 6. Hugging Face backend and remote execution

Use Hugging Face's cache inspection and deletion strategy APIs, with an adapter
around version-dependent capabilities. Validate every selected repository and
every planned path before executing a strategy. Re-scan afterwards: in the
currently installed version some deletion errors are logged rather than raised,
so a successful return from `execute()` alone is not proof of removal.

The first engineering spike must establish a tested capability matrix:

| Layout/operation | Required behavior |
| --- | --- |
| Repository-local blobs, snapshots, refs | List and delete repository/revision scopes while preserving surviving references. |
| Blobs shared across repositories | Correct accounting and preserve blobs still referenced elsewhere; unknown formats block deletion. |
| Malformed cache, missing blobs, interrupted download | Visible diagnosis; no broad recursive deletion fallback. |
| Selective GGUF file removal | Optional later capability; explicit rejection until tested support is available. |

The repo pin is 1.8.0, while newer releases/documentation describe additional
shared-store and file-deletion behavior. Select and record an exact compatible
backend version during the spike. Prefer updating the project's existing pin
after running its compatibility tests; if that cannot be done within scope,
isolate a pinned worker dependency and document the split. Shared-store removal
support is a release gate for advertising support of that cache layout. Never
delegate deletion to whichever unversioned `hf` happens to be on the remote PATH.

Execute the same worker source locally and over `run_remote_script` / existing
SSH stdin sessions. Use JSON input/output with a protocol version, strict field
validation, size limits, and SparkRun's shell-quoting helpers. Bounded parallelism
is appropriate for independent probes; long scans need suitable timeouts and
progress rather than the aggregate probe's fixed 15-second budget.

Tooling readiness is managed by SparkRun:

1. Prefer an existing Python environment with the tested backend capability.
2. Prepare a private, pinned environment through the existing setup machinery;
   add `sparkrun setup model-storage --cluster NAME` for upgrading existing
   clusters. Future normal setup and model preparation provision it as needed.
3. Keep helper preparation separate from read-only commands. If it is missing,
   show an actionable readiness error; never silently install software during
   `list`, `show`, or removal `--dry-run`.
4. Support offline reuse of a prepared environment and preparation using locally
   supplied/cached wheels for nodes without internet access. Do not require a
   globally installed `hf` or a full SparkRun installation on each worker.

The helper must refuse incompatible protocol/backend combinations, unknown cache
formats, and insufficient filesystem permissions. Do not automatically apply
sudo/chown as part of a storage inspection or deletion.

Primary API references, to be checked against the selected pin:
[Hugging Face cache API](https://huggingface.co/docs/huggingface_hub/en/package_reference/cache)
and [cache layout guide](https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache).

## 7. Usage tracking and concurrency

### Durable information

Record preparation receipts independently from job metadata, keyed by physical
cache, repository, and resolved revision. Include recipe reference/fingerprint,
primary/dependency role, exact destinations, SparkRun version, and timestamps.
Store only storage-relevant metadata. Publish receipts atomically after verified
preparation, and preserve them when jobs stop.

Keep target-side receipts/reservations under the reserved cache metadata
directory, and a control-side association index under
`<sparkrun-cache>/model-storage/`. Target-side state lets another controller
identify a managed consumer; the local index is a convenience, not an ownership
authority. Use versioned, atomically written records with permissions restricted
to the configured cache users; neither location stores an HF token.

Receipts enrich a live filesystem scan; they never replace it. Missing receipts
do not hide existing files, and stale receipts do not prove an artifact exists.
Recipe changes retain historical associations without redirecting removal to a
new or old repository automatically.

For usage checks, combine live status across supported executors, persisted job
metadata, target-side launch reservations, and observed mount/process information
where needed. Add a non-mutating pending-operation read: the current reader
prunes stale files, which is inappropriate for a strict deletion preview.
Likewise, add a read-only status option or use the underlying executor query:
the current `api.status()` records a completion snapshot. Storage inspection
must not rewrite that snapshot or trigger configuration migration/bootstrap
writes just to discover existing artifacts.

Unknown or conflicting usage evidence blocks removal of affected targets. A
running workload from an older version without precise receipts should be
matched conservatively using its saved recipe and mounts; missing metadata must
not become an assumption of non-use.

### Coordination protocol

Introduce actual cache coordination shared by download, distribution, launch,
prefetch, and removal. Cover both source and destination caches during transfers.

- A cache-level reader/writer guard coordinates mutations. Preparations and
  launches hold a shared guard; removal takes an exclusive guard. Repository
  mutation locks additionally serialize competing downloads to the same repo.
  Whole-cache exclusion for removal is intentional because blobs can be shared
  across repositories.
- Use POSIX advisory file locks on the actual cache storage for the initial
  Linux implementation, held by local workers or persistent SSH worker sessions.
  Test exclusion with separate processes/clients, not mocked lock objects. Loss
  of a guard session cancels dependent preparation/launch work; its durable
  reservation remains until reconciliation establishes what actually happened.
- Acquire all distinct cache guards in a deterministic global order, with bounded
  waits and clean release on failure. Verify cross-client lock behavior on shared
  filesystems. If reliable exclusion cannot be established, deletion is disabled
  there until a supported coordinator is provided.
- Hold launch protection from before its first model access until a durable
  target-side reservation and observable workload state cover the running job.
  There must be no gap between finishing the download and starting the process.
- Target-side reservations survive a controller crash. Reclaim them only after
  verifying operation/process identity and the absence of a live workload or
  transfer. A timeout, expired informational marker, or reused PID is not enough.
- All current `run`, `proxy load`, benchmark, and strategy-driven launch paths
  using managed host caches must participate before `remove` is enabled. Unsupported
  strategies/backends report a capability limitation rather than bypassing the
  guard.
- After taking exclusive guards, re-check live usage and the cache fingerprint,
  then reconstruct the deletion plan. If selection, paths, refs, or sharing have
  materially changed, abort and require a fresh preview/confirmation. With
  `--yes`, still abort a stale supplied plan; do not silently widen it.

Standalone external Hugging Face processes and older SparkRun clients do not
honor this new protocol. Inspect identifiable active downloads/consumers and
document the coordination boundary. Do not claim a global guarantee for arbitrary
processes or unconfigured clients of an NFS export. On a shared cache, checks must
cover known consumer hosts even if only one host performs the deletion; an
unreachable known consumer blocks it.

## 8. Removal execution and failure handling

The operation has these stages:

1. Resolve scope and selectors; perform read-only scans and usage checks.
2. Build a preview containing exact revisions, variants, locations, sharing,
   estimated savings, and blockers. Prompt in the CLI.
3. Validate readiness/permissions and acquire all required exclusive cache guards.
   If any precondition fails, release them and delete nothing.
4. Revalidate the approved plan and live usage under those guards.
5. Execute the validated Hugging Face strategy once per physical cache, keeping
   guards held. Record per-cache progress and outcomes.
6. Re-scan, verify target removal and surviving shared references, and update
   receipts. Report a structured result and release guards.

Path checks must reject traversal, invalid repository names, symlinked repository
roots escaping the selected cache, unexpected shared-blob destinations, and
replacement of checked paths before mutation. A legitimate symlinked cache root
may be canonicalized and approved once; its identity must remain stable. Verify
the complete deletion strategy, including shared blobs, rather than validating
only the top-level repository path. Never delete an HF home, Hub root, `.locks`
tree, token file, dataset, or space as a side effect of model removal.

Maintain a minimal operation journal for deletion attempts so loss of the SSH
reply is distinguishable from a confirmed failure. On disconnect, mark the result
`indeterminate`, reconnect and reconcile when possible, and prevent new mutations
from overlapping any still-running helper. An interrupted library deletion may
leave a damaged cache; surface it for diagnosis instead of claiming rollback.

Initial CLI exit codes: `0` for complete success/no-op; `1` for blocked, failed,
partial, or incomplete operations; `2` for invalid CLI usage; `130` for interruption.
Structured result/error codes provide finer distinctions. For an incomplete
inventory, print reachable results and return `1` with explicit coverage.

## 9. Implementation sequence

Implement on feature branches off `develop-next`; PRs target `develop-next`, per
the developer guide. The current checkout is only the reference for this plan.

| Step | Work and code areas | Completion gate |
| --- | --- | --- |
| 1. Backend and compatibility spike | Build tiny cache fixtures; establish the HF capability/version matrix; validate remote helper packaging and shared-filesystem exclusion. Record the chosen pin and supported layouts. | Demonstrate correct scans and deletion plans for supported cache layouts; unsupported layouts return explicit blockers. |
| 2. Pure resolution and inventory | Add API types, cache location/identity resolution, pure runtime model dependency hooks, local/remote worker, formatting, `model list/show`, JSON, and helper setup. Refactor shared path resolution under parity tests. | Existing cached models appear on both nodes, including those with no job/recipe records; read-only commands never mutate model/cache state. |
| 3. Preparation receipts and guards | Add durable receipts/reservations and actual coordination; integrate launches, transfers, benchmarks, proxy activation, and supported strategy paths. | Race tests show no deletion can overlap an updated managed launch or transfer; old/unknown usage is surfaced conservatively. |
| 4. Removal planning and execution | Add `plan_remove/remove`, repository/revision selection, confirmation, `--dry-run`, shared-store handling, journals, verification, and the `remove` command. | Safety and failure matrix passes, including partial failures and preservation of shared files. No destructive CLI is enabled before step 3. |
| 5. Model prefetch | Extract model-only preparation using existing transfer primitives; implement `plan_pull/pull`; integrate it with launch so both resolve the same artifacts. | Prefetch works with busy GPUs, honors recipe dependencies/targets/revisions, and does not launch workloads or prepare images. |
| 6. Documentation and release checks | Add `docs/MODELS.md`, README examples, recipe lifecycle notes, proxy-unload retention notes, changelog, and cache-missing/setup guidance. Verify packaging and supported Python versions. | Documented examples match CLI help; all required CI checks and the acceptance scenario below pass. |

Inventory can land independently. Removal and its launch/download coordination
must ship together. Prefetch completes the planned group; a web UI is a separate
consumer of the completed API, not a prerequisite for the CLI release.

## 10. Test plan

All automated deletion tests operate on temporary fixture caches. Mock transport
boundaries, but run the actual worker, filesystem operations, and coordination
protocol in subprocesses where their real behavior matters. Follow existing
`tests/conftest.py` isolation and shell-fixture patterns.

| Area | Required cases |
| --- | --- |
| Inventory | Empty/missing cache; several repos/revisions; removed recipes; incomplete and corrupt repos; permission failure; zero-byte files; escaping symlinks; no network during scans. |
| Accounting | Shared blobs within/between repos; hardlinks; symlink chains; duplicate local/remote aliases; shared exports; different caches with identical paths; unknown totals clearly labeled. |
| Resolution | Local vs remote homes; SSH user; default cluster; explicit hosts; recipe/CLI/cluster cache overrides; head-derived historical paths; HF home vs direct Hub root; pre-placed models; missing probe response; untrusted recipe path overrides. |
| Selection | Exact repository only; local ref resolution; differing refs across hosts; pinned/unpinned recipe; draft model with a different revision; opt-in dependency removal; absent repo no-op; invalid revision failure. |
| GGUF | Multiple quants and shards in one revision; projector files; quant-specific removal rejected unless the backend capability is implemented; whole-repo preview lists its broader effect. |
| Deletion integrity | Shared surviving files retained; library logs a permission failure; target paths replaced after preview; malicious repo/path input; identical commit IDs appearing in different repositories cannot broaden selection. |
| Usage | Active Docker and native jobs; preparing job; stopped job; missing old metadata; externally launched identifiable consumer; unknown/unreachable consumer; cold proxy binding whose activation races removal. |
| Concurrency | Two controllers; pull vs remove; rsync source/destination vs remove; cached launch vs remove; repository already absent; deterministic multi-cache lock order; crash/disconnect during launch and deletion; PID reuse. |
| Multi-host failures | One failed preflight prevents all deletes; failure after one successful delete reports partial; reply loss reports indeterminate; retry reconciles completed work; shared storage is deleted only once. |
| Read-only guarantees | List/show and both planning operations do not delete stale markers, write completion snapshots, create cache identities, install helpers, invoke hooks, migrate config, modify proxy config, or repair permissions. |
| Prefetch | Local/head/per-node modes; selected target indices; shared-cache preferences; independently pinned dependencies; offline cache hit; gated-repo credential error; resumes failed transfers; no GPU scheduling or workload launch. |
| CLI/API | Aliases; target mutual exclusion; noninteractive confirmation; `--yes` cannot bypass blockers; stable JSON and exit status; missing helper; incompatible protocol; API emits no console output. |
| Packaging/compatibility | Worker source included in wheel; supported Python 3.12/3.13; backend pin verified locally/remotely; install does not depend on a globally exposed `hf` executable. |

Suggested new test modules: `test_model_storage.py`, `test_model_storage_worker.py`,
`test_model_storage_locks.py`, `test_api_model.py`, `test_cli_model.py`, and
`test_model_preparation.py`. Extend existing download, distribution, job metadata,
GGUF, CLI completion, runtime dependency, launch, and proxy tests where those
paths change.

Run targeted tests while implementing, then the repository's required pytest and
Ruff checks through its normal CI matrix. Verify the wheel contains the worker
and `python scripts/vendor-sparkroute.py verify` still passes; no vendored gateway
changes are needed for this feature. Follow `versions.yaml` generation rules if
version metadata changes.

## 11. Acceptance scenario

Use a tiny test repository or synthetic cache in dedicated test directories;
never choose a production model for deletion validation.

1. Inventory two nodes with different local copies, plus a separate control
   cache. Show per-location usage and paths.
2. Add two recipe fixtures referencing the same primary repo, with one requiring
   a separately pinned draft repo. Prefetch the latter without launching anything.
3. Verify a subsequent launch reuses the prepared files. Start a consuming test
   workload and confirm `remove` refuses it.
4. Stop/unload that workload. Confirm both `model list` and its receipts still
   show the weights.
5. Preview removal. Confirm exact targets, revision scope, other recipe
   associations, control-cache exclusion, and estimated reclaimable bytes.
6. Remove the primary repo from the selected nodes. Verify its files are absent,
   unrelated/shared resources survive, and the control copy remains unless
   explicitly selected.
7. Repeat against shared storage and a simulated mid-operation host failure.
   Verify deduplication, clear partial outcomes, and an idempotent retry.
8. Remove a recipe definition before inspecting another cached repo; it must
   remain discoverable/removable by model ID. Restore/run the original recipe
   and confirm deleted weights are fetched again.

## 12. Follow-up capabilities

After the command group and API meet the acceptance gates:

- Add selective GGUF/file removal with reference-aware shared-blob handling.
- Add explicit retention pins and previewable pruning of old revisions or
  interrupted downloads. A model being stopped or absent from a proxy is not
  sufficient evidence that the user wants its files evicted.
- Add a Storage page using the same inventory, removal plan, and result records;
  any HTTP surface must enforce authentication/authorization and server-side
  selection validation independently of the CLI.
- Add provider-specific storage adapters for managed volumes.
- Consider a composed stop-and-remove convenience operation once the separate
  lifecycle operations and their failure behavior are established.
