"""Activation offloading: tensors autograd saves (outside checkpointed layers, i.e. each
checkpointed layer's input) go to pinned CPU memory and come back for the backward.

torch's non-reentrant checkpoint stores a layer's inputs as saved tensors, so they pass
through these hooks; inside a checkpointed layer the checkpoint's own hooks take over.
Pinned buffers are pooled by size and reused every step (allocating pinned memory is
slow), and freed back to the pool once the step's backward is done.
"""
import torch

MIN_BYTES = 1 << 20  # smaller tensors stay on the GPU


class ActivationOffload:
    def __init__(self, device):
        self.device = device
        self.pool = {}      # (numel, dtype) -> [free pinned buffers]
        self.in_use = []    # buffers handed out this step
        self.stream = torch.cuda.Stream(device) if device.type == "cuda" else None

    def _buffer(self, t):
        key = (t.numel(), t.dtype)
        free = self.pool.setdefault(key, [])
        buf = free.pop() if free else torch.empty(t.numel(), dtype=t.dtype, pin_memory=True)
        self.in_use.append((key, buf))
        return buf

    def pack(self, t):
        if t.device.type != "cuda" or t.numel() * t.element_size() < MIN_BYTES or not t.is_contiguous():
            return t
        buf = self._buffer(t)
        if self.stream is not None:
            # copy on a side stream, after the producer's work; the GPU tensor stays alive until done
            self.stream.wait_stream(torch.cuda.current_stream(t.device))
            with torch.cuda.stream(self.stream):
                buf.copy_(t.view(-1), non_blocking=True)
                ev = torch.cuda.Event()
                ev.record(self.stream)
            t.record_stream(self.stream)
        else:
            buf.copy_(t.view(-1))
            ev = None
        return (buf, t.shape, t.device, ev)

    def unpack(self, packed):
        if isinstance(packed, torch.Tensor):
            return packed
        buf, shape, device, ev = packed
        if ev is not None:
            torch.cuda.current_stream(device).wait_event(ev)
        return buf.to(device, non_blocking=True).view(shape)

    def context(self):
        return torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack)

    def end_step(self):
        """After the backward: every buffer of this step can be reused."""
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()
        for key, buf in self.in_use:
            self.pool[key].append(buf)
        self.in_use = []

    def pinned_bytes(self):
        return sum(b.numel() * b.element_size() for bs in self.pool.values() for b in bs) + \
            sum(b.numel() * b.element_size() for _, b in self.in_use)
