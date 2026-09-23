# torch-numpy-stream-shuffle-guard

This repository has been consolidated into [`torch-correctness-guards`](https://github.com/zhuhroscar-tech/torch-correctness-guards).

Use the umbrella package instead:

```bash
python -m pip install "torch-correctness-guards[torch]"
torch-guard run numpy-stream-shuffle
```

Python API:

```python
from torch_correctness_guards import safe_row_shuffle, safe_row_shuffle_
```

The original guard functionality is preserved as the `numpy-stream-shuffle` module/subcommand in the umbrella package. This repo is kept as an archival pointer only.
