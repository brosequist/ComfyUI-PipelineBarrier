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


class _AnyType(str):
    """Sentinel that compares equal to every ComfyUI type string.

    Allows PipelineMemoryBarrier to sit on any wire in the graph without
    requiring a specific type declaration.
    """

    def __ne__(self, other: object) -> bool:
        return False


_ANY = _AnyType("*")


def _log_memory(label: str) -> None:
    n = torch.cuda.device_count()
    for i in range(n):
        alloc = torch.cuda.memory_allocated(i) / 1024**3
        rsvd = torch.cuda.memory_reserved(i) / 1024**3
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
    """Force-flush GPU caches and run GC between pipeline stages.

    Place this node between pipeline stages to ensure GPU and CPU memory is
    fully released before the next stage loads its models.  The input value
    passes through unchanged, so the node is non-destructive and can be
    inserted on any wire.

    Inputs
    ------
    passthrough : any
        Any value — LATENT, MODEL, IMAGE, CONDITIONING, etc.  Returned
        unchanged as the only output.
    log_memory : bool
        When True (default), prints GPU allocated/reserved and system RAM
        usage before and after the flush to the ComfyUI console.

    Outputs
    -------
    passthrough : same type as input
        The unmodified input value.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "passthrough": (_ANY,),
            },
            "optional": {
                "log_memory": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = (_ANY,)
    RETURN_NAMES = ("passthrough",)
    FUNCTION = "barrier"
    CATEGORY = "utils/memory"
    OUTPUT_NODE = False

    def barrier(self, passthrough, log_memory: bool = True):
        if log_memory:
            _log_memory("before flush")
        _flush()
        if log_memory:
            _log_memory("after  flush")
        return (passthrough,)


NODE_CLASS_MAPPINGS = {
    "PipelineMemoryBarrier": PipelineMemoryBarrier,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PipelineMemoryBarrier": "Pipeline Memory Barrier",
}
