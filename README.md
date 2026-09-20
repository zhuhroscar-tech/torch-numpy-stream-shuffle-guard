# torch-numpy-stream-shuffle-guard

Detects and guards against a real, currently-open PyTorch correctness
defect: `torch._numpy.random.shuffle` on a `torch.Tensor`, under
`torch._dynamo.config.use_numpy_random_stream=True`, does **not** perform
a true in-place permutation of the tensor's rows — it can **duplicate
some rows and silently drop others**, so the result is not a permutation
of the input at all.

Tracking issue: [pytorch/pytorch#197795](https://github.com/pytorch/pytorch/issues/197795)
(open, no fix merged as of this guard's creation).

## The bug, in one example

```python
import torch
import torch._numpy as tnp

x = torch.arange(12).reshape(6, 2)
tnp.random.seed(0)
with torch._dynamo.config.patch(use_numpy_random_stream=True):
    tnp.random.shuffle(x)

print(x.tolist())
# [[0, 1], [2, 3], [2, 3], [0, 1], [4, 5], [4, 5]]
# rows [0,1] and [2,3] each appear TWICE; two other rows vanished entirely.
```

Independently reproduced on this project's own CI and on the
maintainer's Apple M4/arm64 host (torch 2.14.0, CPU-only): a real
`UserWarning` is emitted ("you are shuffling a 'Tensor' object which is
not a subclass of 'Sequence'... may contain duplicates after
shuffling"), but the call does **not** raise — the caller silently
receives a corrupted tensor.

**Root cause** (per the upstream issue): stream-mode shuffle delegates
to NumPy's own `Generator.shuffle`/`RandomState.shuffle`, whose in-place
Fisher-Yates loop assumes real NumPy ndarray view/stride semantics. A
`torch.Tensor` is not a `Sequence` and does not share those exact
aliasing semantics, so the swap loop silently corrupts the tensor's
backing storage instead of performing a genuine permutation.

## The guard

`safe_row_shuffle(x)` / `safe_row_shuffle_(x)` never call the unsafe
stream-mode path. They perform a real permutation via `torch.randperm`
(a native, tensor-safe RNG op with no view-aliasing hazard) followed by
`index_select` — the standard, already-recommended way to permute a
tensor's leading axis in PyTorch. Because it is a pure gather over a
bijective index permutation, the output's row multiset is **identical**
to the input's by construction; there is no code path by which this
implementation can duplicate or drop a row.

```python
from torch_numpy_stream_shuffle_guard.core import safe_row_shuffle_

x = torch.arange(12).reshape(6, 2)
safe_row_shuffle_(x)  # true in-place row permutation, multiset preserved
```

## Install

```bash
pip install torch-numpy-stream-shuffle-guard[torch]
```

The `torch` extra is optional at install time (the CLI degrades
gracefully with a clear JSON error and exit code 2 if torch is not
installed) but required to actually run the diagnosis or guard
functions.

## CLI usage

```bash
torch-numpy-stream-shuffle-guard            # human-readable report
torch-numpy-stream-shuffle-guard --json     # machine-readable report
torch-numpy-stream-shuffle-guard --no-color # disable ANSI color
torch-numpy-stream-shuffle-guard --version
```

Exit code is `0` if the guard is fully effective on this host's
installed torch build, `1` if the guard itself failed to preserve the
row multiset in any tested case (would indicate a defect in this
project, not upstream), and `2` if torch is not installed.

## Limitations (stated honestly)

- This guards only the **row-permutation-via-shuffle** failure mode
  described in pytorch/pytorch#197795. It does not attempt to fix or
  wrap every other `torch._numpy.random` function.
- `safe_row_shuffle` only permutes along axis 0 (rows). It does not
  attempt an axis-arbitrary shuffle.
- Verified on CPU only (Apple Silicon arm64 and Linux x86_64 CI); not
  tested against CUDA/MPS-resident tensors, though `torch.randperm` +
  `index_select` are standard, well-supported ops on those backends too.
- This is a workaround for a real, currently-open upstream bug, not an
  upstream fix. If pytorch/pytorch#197795 is fixed upstream, re-verify
  `diagnose()`'s `any_bug_present` field against the newly installed
  torch version — the test suite is written to flag (not silently pass)
  if the bug disappears on this host's build, so a future run's CI
  failure on that specific test is the expected signal to revisit this
  README and the guard's continued necessity.

## License

MIT
