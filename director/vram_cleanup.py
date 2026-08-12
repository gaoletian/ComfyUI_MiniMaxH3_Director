"""Release GPU memory between MiniMax H3 Director segment runs."""

from __future__ import annotations

import gc
import logging

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.vram")


def cleanup_segment_vram(*, enabled: bool = True, unload_models: bool = True) -> None:
    """Release segment GPU memory: gc, optional unload of ComfyUI models, empty CUDA cache."""
    if not enabled:
        return
    gc.collect()
    try:
        import comfy.model_management as mm

        mm.cleanup_models_gc()
        if unload_models:
            mm.unload_all_models()
            mm.cleanup_models()
        mm.soft_empty_cache()
    except Exception as exc:
        log.warning("Segment VRAM cleanup failed: %s", exc)
        return
    if unload_models:
        log.debug("MiniMax H3 Director: segment VRAM cleanup (models unloaded, cache cleared)")
    else:
        log.debug("MiniMax H3 Director: segment VRAM cleanup (cache cleared, models kept loaded)")


def release_director_run(*, unload_models: bool = True) -> None:
    """Reclaim run-scoped memory after a cancelled or failed director run.

    A ComfyUI cancel surfaces as ``InterruptProcessingException`` (a
    ``BaseException``) from inside the sampler; the prompt executor catches it
    but only runs ``cleanup_models_gc()`` — it never calls ``soft_empty_cache``
    or ``unload_all_models``. The freed AV latents / conditioning tensors
    therefore stay reserved in the CUDA caching allocator and the loaded UNET /
    VAE / CLIP stay resident. This helper makes the reclaim deterministic on any
    abnormal exit (interrupt, error, KeyboardInterrupt): gc + unload models +
    empty device cache, and drops module-level preview caches that a cancel
    would otherwise leave in VRAM.
    """
    try:
        from .tae_preview import release_tae_decoder

        release_tae_decoder()
    except Exception as exc:
        log.debug("TAE preview decoder release skipped: %s", exc)
    cleanup_segment_vram(enabled=True, unload_models=unload_models)
    log.info("MiniMax H3 Director: run cancelled/aborted — memory reclaimed")
