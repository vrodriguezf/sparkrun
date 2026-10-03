"""Model-only dependency and transfer seam shared with inference launches."""

from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
from contextvars import ContextVar

quiet_preparation = ContextVar("quiet_model_preparation", default=False)


@contextmanager
def quiet_model_preparation():
    """Library callers never inherit CLI-dependent terminal streaming."""
    token = quiet_preparation.set(True)
    try:
        yield
    finally:
        quiet_preparation.reset(token)


def model_inputs(recipe, runtime=None):
    """Resolve on a copy; each distribution entry's revision is authoritative."""
    recipe = deepcopy(recipe)
    recipe.distribution_config.resolve(recipe)
    if runtime is not None:
        for entry in runtime.model_dependencies(recipe):
            recipe.distribution_config.add_model(entry.name, revision=entry.revision)
    # Serving always consumes the primary model, even if a hand-written
    # distribution list mentions only auxiliary inputs.
    if not any(entry.name == recipe.model for entry in recipe.distribution_config.models.entries):
        recipe.distribution_config.add_model(recipe.model, revision=recipe.model_revision)
    return tuple(recipe.distribution_config.models.entries)


def prepare_model(
    model,
    targets,
    *,
    full_hosts,
    cache_dir,
    local_cache_dir,
    transfer_mode,
    ssh_kwargs,
    revision=None,
    hf_token=None,
    prefs=None,
    dry_run=False,
):
    """Use the launcher's weights-only transfer implementation (no image/IB work)."""
    from sparkrun.orchestration.distribution import _distribute_single_model

    return _distribute_single_model(
        model,
        list(targets),
        list(full_hosts),
        cache_dir,
        local_cache_dir,
        transfer_mode,
        None,
        None,
        ssh_kwargs,
        revision,
        hf_token,
        dry_run,
        False,
        prefs=prefs,
    )
