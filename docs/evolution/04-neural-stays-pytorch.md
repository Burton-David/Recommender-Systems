# Phase 4 — TwoTowerCF stays in PyTorch

**Status:** shipped.
**Tag:** `v0.4.0` (planned).
**Code change:** none. This phase is a deliberate refusal documented as
an ADR, plus the version bump.

## Context

Phases 2 and 3 each took an algorithm whose hot path the
[Phase 1.1 measurements](01-1-bench-baseline.md) had named —
[BPR's `_step` to Rust](02-rust-kernel-bpr.md),
[k-NN and SVD to sparse](03-sparse-recommenders.md) — and rewrote it. The
pattern is set; the obvious next move would be to keep going and rewrite
`TwoTowerCF` too. Faster training? Less Python overhead? Tighter coupling
between the library's compiled paths?

**No.** This phase exists to argue the discipline of *not* doing the
rewrite, and to make that argument durable enough that a future
contributor — or future me — doesn't quietly reopen the question without
re-reading what was already considered.

## What `TwoTowerCF` actually is

A small `nn.Module` with two embedding tables and a dot-product head,
trained by `torch.optim.Adam` on a BPR-style ranking loss. Per
`src/recommender_systems/neural.py`:

```python
class _TwoTowerNet(nn.Module):
    def __init__(self, n_users, n_items, n_factors):
        ...
        self.user_embed = nn.Embedding(n_users, n_factors)
        self.item_embed = nn.Embedding(n_items, n_factors)

    def forward(self, users, items):
        return (self.user_embed(users) * self.item_embed(items)).sum(dim=-1)
```

The numerical work — embedding lookups, the dot product, the
log-sigmoid loss, the backward pass — is already a thin Python wrapper
over PyTorch's ATen C++ kernels (and CUDA kernels on GPU). The Python
interpreter is involved per *batch*, not per (user, positive, negative)
triple the way it was in pure-numpy BPR.

That difference is the whole reason to leave it alone.

## Decision

Don't rewrite `TwoTowerCF`. Keep it as the PyTorch reference
implementation it already is.

## Options considered

| Option | Rejected because |
|---|---|
| **Rewrite the inner loop in Rust** (the same kind of move that worked for `BPR._step`) | BPR's Python loop was 500,000 interpreter calls on tiny vector ops — the interpreter was the bottleneck. `TwoTowerCF` calls into the interpreter ~`len(positives)/batch_size` times per epoch — i.e., hundreds, not hundreds of thousands — and each call hands off to compiled C++/CUDA. There's no equivalent interpreter overhead to eliminate. |
| **Port to `tch-rs`** (Rust bindings to LibTorch) | Same underlying ATen kernels, same speed, lose Python autograd ergonomics, lose the entire pip-installable PyTorch ecosystem (Hugging Face checkpoints, `torch.compile`, `torch.export`, mixed precision, distributed training). A Rust user of this library still needs Python somewhere; multiplying that surface for no perf gain isn't a trade we make. |
| **Hand-rolled CUDA kernels** | PyTorch's stock `Embedding` + matmul + log-sigmoid kernels are already well-tuned. Beating them takes a specific bottleneck — something like custom embedding-bag fusion for ranking — which we don't have. Premature optimization with significant build-system cost (CUDA toolkit in CI, GPU runners, version-pinning headaches). |
| **JAX rewrite** | TPU-native speedups are real on Google Cloud TPUs. Adding a JAX dep splits the codebase across two ML frameworks, doubles the testing burden, and asks users to install JAX-on-CPU for a feature that mostly benefits Google-Cloud TPU users. Not worth it for this library's audience. |
| **Replace it with `torch.compile`'d eager training** | Considered as an in-place optimization. `torch.compile` over the existing `_TwoTowerNet.forward` would help on long epochs, but adds compile-time overhead that dominates on the short epoch budgets we target in tests, and requires PyTorch ≥ 2.1 as a hard floor instead of the current ≥ 2.0. The cost-benefit isn't there yet. Worth revisiting if the neural module grows. |

## What "TwoTowerCF stays in PyTorch" actually means

Several things explicitly NOT true:

- It does **not** mean PyTorch is special or untouchable. The next ML
  framework that better matches the workload would be considered on
  the same grounds.
- It does **not** mean the neural path won't get optimization work.
  When `torch.compile`'s overhead stops being painful or when a
  measurable bottleneck appears, that's a future phase. The principle
  is *measured before cut*, not "PyTorch is fine forever."
- It does **not** mean the neural module can't grow. Side-information
  towers, attention, sequence-aware variants — those are all clean
  extensions of the existing PyTorch code. Doing them in Python +
  PyTorch is the path of least resistance and the path most aligned
  with where recsys research actually publishes.

## What this phase is, in one sentence

The repo's pattern through Phase 3 was "find the slowest thing the data
points at, rewrite it in whatever compiled language matches the
workload." `TwoTowerCF` is already on a compiled C++/CUDA path through
PyTorch — applying the same pattern would mean swapping compiled-via-
PyTorch for compiled-via-Rust-bindings-to-PyTorch, which is the same
underlying kernels with a worse ecosystem story. Refusing the rewrite
*is* the engineering decision, and writing it down is what makes the
refusal durable.

## What's not in scope

- Any code change to `src/recommender_systems/neural.py`.
- Any change to the neural extra (`pip install -e ".[neural]"`).
- Any benchmark on goodbooks-full for the neural model. The neural
  path's scaling story is its own question (GPUs, distributed
  training, sequence-aware variants); separate from "should we
  rewrite the inner loop in Rust."
