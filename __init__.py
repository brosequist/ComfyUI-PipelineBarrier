"""
ComfyUI-PipelineBarrier — force-flush GPU caches between pipeline stages.

Inserts an explicit GPU cache flush and GC pass into a ComfyUI workflow graph.
Useful in multi-stage or multi-GPU pipelines (e.g. WanVideo two-pass sampling)
where model offload alone does not fully release GPU and CPU memory before the
next stage begins, leading to OOM kills.
"""

import gc

import psutil
import torch

import comfy.model_management as mm


def _log_memory(label: str) -> None:
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1024**3
        rsvd  = torch.cuda.memory_reserved(i)  / 1024**3
        print(
            f"[PipelineBarrier] {label} | GPU {i}: "
            f"alloc={alloc:.2f} GiB  reserved={rsvd:.2f} GiB"
        )
    vm = psutil.virtual_memory()
    print(
        f"[PipelineBarrier] {label} | RAM: "
        f"{vm.used / 1024**3:.1f}/{vm.total / 1024**3:.1f} GiB  ({vm.percent:.0f}% used)"
    )


def _flush() -> None:
    mm.soft_empty_cache(force=True)
    gc.collect()
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


class PipelineMemoryBarrier:
    """Force-flush GPU caches between pipeline stages.

    Pass a LATENT through this node to trigger an explicit GPU cache flush
    before the next stage loads its models.  Zero compute cost; useful between
    samplers and before VAE decode in multi-stage workflows.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
            },
            "optional": {
                "log_memory": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES  = ("LATENT",)
    RETURN_NAMES  = ("latent",)
    FUNCTION      = "barrier"
    CATEGORY      = "utils/memory"
    OUTPUT_NODE   = False

    def barrier(self, latent, log_memory: bool = True):
        if log_memory:
            _log_memory("before flush")
        _flush()
        if log_memory:
            _log_memory("after  flush")
        return (latent,)


NODE_CLASS_MAPPINGS         = {"PipelineMemoryBarrier": PipelineMemoryBarrier}
NODE_DISPLAY_NAME_MAPPINGS  = {"PipelineMemoryBarrier": "Pipeline Memory Barrier"}
