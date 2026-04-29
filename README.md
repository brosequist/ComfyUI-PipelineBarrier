# ComfyUI-PipelineBarrier

A single-node ComfyUI extension that inserts an explicit GPU cache flush and garbage-collection pass between pipeline stages.

## Why?

Long multi-stage workflows — especially [WanVideo](https://github.com/Kosinkadink/ComfyUI-WanVideoWrapper) two-pass I2V with [ComfyUI-MultiGPU](https://github.com/city96/ComfyUI-MultiGPU) — can OOM-kill the process at the start of the next stage even after the previous model has been offloaded via BlockSwap's `force_offload`. The problem is that PyTorch's CUDA allocator caches freed blocks rather than returning them to the OS immediately, so the system sees less free RAM than is actually available.

`PipelineMemoryBarrier` calls ComfyUI's `soft_empty_cache(force=True)`, Python's `gc.collect()`, and PyTorch's `torch.cuda.empty_cache()` + `torch.cuda.synchronize()` on every visible GPU, ensuring caches are flushed and memory is returned to the OS before the next stage begins.

## Installation

### Via ComfyUI Manager (recommended)

Search for **Pipeline Memory Barrier** in the ComfyUI Manager node list.

### Manual

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/brosequist/ComfyUI-PipelineBarrier
```

No pip dependencies beyond what ComfyUI already installs (`psutil` is included in the base environment).

## Usage

The node accepts any input type and returns it unchanged, so it can be inserted on any wire.

**Typical placement in a WanVideo two-pass workflow:**

```
[Pass 1 Sampler] --latent--> [Pipeline Memory Barrier] --latent--> [Pass 2 Sampler]
[Pass 2 Sampler] --latent--> [Pipeline Memory Barrier] --latent--> [VAE Decode]
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `passthrough` | any | — | Any value. Returned unchanged. |
| `log_memory` | bool | `True` | When enabled, logs GPU allocated/reserved and system RAM before and after the flush to the ComfyUI console. |

### Example console output

```
[PipelineBarrier] before flush | GPU 0: alloc=14.32 GiB  reserved=15.12 GiB
[PipelineBarrier] before flush | GPU 1: alloc=7.18 GiB   reserved=8.00 GiB
[PipelineBarrier] before flush | RAM: 41.2/62.8 GiB  (66% used)
[PipelineBarrier] after  flush | GPU 0: alloc=0.01 GiB   reserved=0.01 GiB
[PipelineBarrier] after  flush | GPU 1: alloc=0.00 GiB   reserved=0.00 GiB
[PipelineBarrier] after  flush | RAM: 11.4/62.8 GiB  (18% used)
```

## Notes

- The node costs essentially zero compute time — it is a synchronization point only.
- `log_memory=False` silences the console output for production workflows.
- The flush is safe to call at any point; it does not unload models that ComfyUI is still tracking, only releases PyTorch's internal cache of freed blocks.
- On multi-GPU setups all visible CUDA devices are flushed.

## License

MIT
