"""Build environments and stage recipe assets without activating inference."""

from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import replace
import logging
from typing import Any

from sparkrun.api._build_models import BuildOptions, BuildPlan, BuildResult
from sparkrun.api._context import resolve_sctx
from sparkrun.api._errors import HostsUnreachable, IntegrationUnavailable, SparkrunError, TrustRejected
from sparkrun.core.model_storage_locks import guarded_operation, reserve_launch

logger = logging.getLogger(__name__)


def plan_build(options: BuildOptions, *, sctx=None) -> BuildPlan:
    """Resolve all preparation hosts, without scheduling ranks or checking VRAM.

    Transport preparation may refresh connection details. No builder, transfer,
    plugin preparation hook, or workload lifecycle operation runs here.
    """
    from sparkrun.api._resolve import resolve_cluster, resolve_recipe, resolve_runtime, scope_operation
    from sparkrun.core.installed_plugins import RequiredIntegrationError, require_integrations
    from sparkrun.core.launcher import _enforce_recipe_mount_trust, resolve_recipe_trust
    from sparkrun.core.validation import validate_for_launch
    from sparkrun.orchestration.executor import resolve_executor

    sctx = resolve_sctx(sctx)
    try:
        require_integrations()
    except RequiredIntegrationError as error:
        raise IntegrationUnavailable(str(error)) from error

    recipe = resolve_recipe(deepcopy(options.recipe), sctx=sctx, overrides=deepcopy(options.overrides))
    if options.rebuild is not None:
        recipe.builder_config["rebuild"] = options.rebuild
    cluster = deepcopy(resolve_cluster(options.cluster, options.hosts, sctx=sctx))
    hosts = tuple(cluster.hosts)
    if not hosts:
        raise HostsUnreachable("Build requires at least one target host.")
    runtime = resolve_runtime(recipe, sctx=sctx)
    if runtime.is_delegating_runtime():
        raise SparkrunError("Build-only preparation is not supported by delegating runtime %r." % runtime.runtime_name)
    sctx, _ = scope_operation(cluster, sctx=sctx, dry_run=options.dry_run)
    try:
        executor = resolve_executor(
            recipe=recipe,
            runtime=runtime,
            cluster=cluster,
            config=sctx.config,
            cli_overrides=options.executor_overrides(),
            host_hardware=cluster.hardware_for(hosts[0]),
            v=sctx.variables,
        )
        if executor.executor_name not in {"docker", "local"}:
            raise SparkrunError("Build-only preparation is not supported by executor %r." % executor.executor_name)
        target = executor.resolve_target(dry_run=options.dry_run)
        target_overrides = target.overrides
        target_overrides.pop("executor", None)
        cluster = replace(cluster, executor=target.executor, executor_config={**(cluster.executor_config or {}), **target_overrides})
        validation_recipe = copy(recipe)
        validation_recipe.executor = target.executor
        issues, failed = validate_for_launch(
            validation_recipe,
            runtime=runtime,
            cluster=cluster,
            config=sctx.config,
            v=sctx.variables,
            overrides=options.overrides,
            include_unmapped_keys=False,
        )
        if failed:
            raise SparkrunError("Build recipe validation failed: " + "; ".join(issue.message for issue in issues))
        _enforce_recipe_mount_trust(recipe, resolve_recipe_trust(recipe, options.trust, sctx=sctx))
    except SparkrunError:
        raise
    except Exception as error:
        raise SparkrunError("Build planning failed: %s" % error) from error
    return BuildPlan(recipe, runtime, cluster, hosts, target, executor.needs_image, options, tuple(issues))


def build(options: BuildOptions, *, plan: BuildPlan | None = None, sctx=None) -> BuildResult:
    """Prepare a recipe's images or native environment and distribute assets.

    A supplied plan is reused without rescheduling or preparing transport again.
    It must be created with the same options. Temporary builder/probe processes
    are allowed; inference startup, snapshot capture/restore, launch hooks,
    eviction, page-cache clearing, and serving-job registration are excluded.
    """
    from sparkrun.core.timing import Timeline, timed

    sctx = resolve_sctx(sctx)
    if plan is None:
        plan = plan_build(options, sctx=sctx)
    elif plan.options != options:
        raise ValueError("build plan must be created with the same BuildOptions")
    sctx = sctx.for_cluster(plan.cluster)
    timeline = sctx.timing or Timeline()
    try:
        with timed(timeline, "build", hosts=len(plan.host_list), dry_run=options.dry_run):
            result = _prepare(options, plan=plan, sctx=sctx, timeline=timeline)
    except SparkrunError:
        raise
    except Exception as error:
        raise SparkrunError("Build preparation failed: %s" % error) from error
    return replace(result, timeline=timeline.export())


@guarded_operation
def _prepare(options: BuildOptions, *, plan: BuildPlan, sctx, timeline) -> BuildResult:
    from sparkrun.core.asset_preparation import prepare_tuning
    from sparkrun.core.build_preparation import BuildContext, prepare_build_context
    from sparkrun.core.cluster_manager import ModelDistributionPrefs
    from sparkrun.core.image_preparation import prepare_images, stage_prepared_images, validate_image_configuration
    from sparkrun.core.launcher import (
        _verify_mount_sources,
        resolve_effective_cache_dir,
        resolve_recipe_trust,
    )
    from sparkrun.core.recipe import is_local_model_path
    from sparkrun.core.recipe_items import registered_recipe_items
    from sparkrun.core.timing import timed
    from sparkrun.orchestration.distribution import _resolve_targets, distribute_from_config, resolve_auto_transfer_mode
    from sparkrun.orchestration.primitives import build_ssh_kwargs

    recipe = deepcopy(plan.recipe)
    runtime = plan.runtime
    cluster = plan.cluster
    hosts = list(plan.host_list)
    config = sctx.config
    ssh_kwargs = build_ssh_kwargs(config)
    overrides = deepcopy(options.overrides)
    trusted = resolve_recipe_trust(recipe, options.trust, sctx=sctx)
    if not trusted and any(r.key in recipe.plugin_items and r.build_preparation for r in registered_recipe_items()):
        raise TrustRejected("Recipe build hooks require explicit authorization: pass BuildOptions(trust=True) (CLI: --trust).")
    if plan.needs_image:
        validate_image_configuration(recipe, runtime, v=sctx.variables)

    cc = recipe.cluster_config
    cache_dir = options.cache_dir or cluster.cache_dir
    local_cache = options.local_cache_dir or str(config.hf_cache_dir)
    resolved_model = ""
    if cc is not None:
        cache_dir = cc.remote_cache_dir or cache_dir
        local_cache = cc.local_cache_dir or local_cache
        resolved_model = cc.resolved_model_path or ""
        if resolved_model:
            recipe.model = resolved_model
    skip_model = bool(resolved_model or is_local_model_path(recipe.model))
    if not options.dry_run:
        _verify_mount_sources(
            recipe,
            hosts,
            ssh_kwargs,
            runtime=runtime,
            cluster=cluster,
            config=config,
            overrides={**overrides, **plan.executor_target.overrides},
        )
    effective_cache = resolve_effective_cache_dir(cache_dir, hosts, ssh_kwargs, config, dry_run=options.dry_run)
    topology = options.topology or cluster.topology
    transfer = resolve_auto_transfer_mode(
        options.transfer_mode or cluster.transfer_mode or "auto",
        hosts,
        ssh_kwargs=ssh_kwargs,
        dry_run=options.dry_run,
        topology=topology,
        mgmt_interface=cluster.mgmt_interface,
    )

    reserve_launch(
        recipe,
        runtime,
        hosts,
        effective_cache,
        local_cache,
        ssh_kwargs,
        include_local=transfer.mode in {"local", "push"} or transfer.auto_delegated,
        dry_run=options.dry_run,
    )

    with timed(timeline, "build.environment"):
        logger.info("Preparing %s environment on %d host(s)", recipe.builder or plan.executor_target.executor, len(hosts))
        context = prepare_build_context(BuildContext(options, plan, recipe, sctx))
        prepared = prepare_images(
            recipe,
            runtime,
            hosts,
            config=config,
            v=sctx.variables,
            cluster=cluster,
            dry_run=options.dry_run,
            transfer_mode=transfer.mode,
            ssh_kwargs=ssh_kwargs,
            needs_image=plan.needs_image,
            builder_context=context,
        )
        runtime.prepare(
            recipe,
            hosts,
            config=config,
            dry_run=options.dry_run,
            transfer_mode=transfer.mode,
            overrides=overrides,
        )

    model_prefs = cluster.distribution.model
    skip_model = skip_model or not model_prefs.enabled
    prefs = ModelDistributionPrefs(
        preserve_perms=options.preserve_model_perms if options.preserve_model_perms is not None else model_prefs.preserve_perms,
        skip_fan_out=options.skip_model_fan_out if options.skip_model_fan_out is not None else model_prefs.skip_fan_out,
    )
    distribution: dict[str, Any] = dict(
        dry_run=options.dry_run,
        recipe_name=recipe.name,
        transfer_mode=transfer.mode,
        transfer_interface=options.transfer_interface or cluster.transfer_interface,
        local_cache_dir=local_cache,
        pre_ib=transfer,
        topology=topology,
        mgmt_interface=cluster.mgmt_interface,
        cluster_name=cluster.name,
        prefs=prefs,
        timeline=timeline,
    )
    with timed(timeline, "build.distribute"):
        if plan.needs_image:
            staged = stage_prepared_images(
                prepared,
                recipe,
                hosts,
                effective_cache,
                config,
                require_content_ids=True,
                stage_models=not skip_model,
                ssh_kwargs=ssh_kwargs,
                **distribution,
            )
            images = staged.content_images_by_node
        else:
            distribute_from_config(
                recipe,
                "",
                hosts,
                effective_cache,
                config,
                skip_container=True,
                skip_model=skip_model,
                **distribution,
            )
            images = ()

    with timed(timeline, "build.tuning"):
        prepare_tuning(
            recipe,
            runtime,
            hosts,
            cluster=cluster,
            registry_mgr=sctx.registry_manager,
            sync_tuning=options.sync_tuning,
            dry_run=options.dry_run,
            transfer_mode=transfer.mode,
            ssh_kwargs=ssh_kwargs,
            strict=True,
        )

    resolved = deepcopy(recipe.distribution_config)
    resolved.resolve(recipe, resolved_container=prepared.head_image or "")
    models = (
        tuple(entry.name for entry in resolved.models.entries if entry.name and _resolve_targets(entry.target or [-1], hosts))
        if not skip_model and resolved.models.enabled
        else ()
    )
    return BuildResult(
        recipe=recipe.qualified_name,
        host_list=plan.host_list,
        executor=plan.executor_target.executor,
        dry_run=options.dry_run,
        container_image=prepared.head_image,
        images_by_node=images,
        models=models,
        effective_cache_dir=effective_cache,
        local_cache_dir=local_cache,
        environment_file=prepared.builder.default_env_file(recipe) if prepared.builder else None,
    )
