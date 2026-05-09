# Slipnet Weight Learner

A hill-climbing optimizer that learns the link weights of Copycat's Slipnet by
minimizing mean final workspace temperature across all 9 analogy problems.

## What was learned

The Copycat Slipnet has 202 directed links.  132 of those have a non-zero
`fixedLength` value (the remaining 70 derive their strength from their label
node at runtime and are treated as structural — never perturbed).

The learner optimizes those 132 `fixedLength` values by running Copycat with
each candidate weight configuration and measuring the mean final temperature
across 9 problems × 5 fixed random seeds = 45 trials per evaluation.

Lower temperature means Copycat found a more confident, coherent answer.
The baseline (original Mitchell weights) achieves a mean temperature of roughly
**23.7** across the 9 problems.

Two problems are tracked as **diagnostics** in every log entry because they
probe particularly interesting slippage behaviour:

| # | Problem | Why diagnostic |
|---|---------|---------------|
| 3 | `abc → abd`, target `kji` | Reverse-alphabet target — tests direction-category slippage |
| 6 | `abc → abd`, target `xyz` | Late-alphabet target — tests successor-chain reasoning |

## How to run

### Prerequisites

```bash
pip install -r copycat_src/requirements.txt   # only 'future' is needed
```

### 1 — Generate the baseline weights (one-time)

```bash
python -c "
import learner
from copycat.slipnet import Slipnet
from learner.weight_space import extract_weights, save
import os; os.makedirs('weights', exist_ok=True)
save(extract_weights(Slipnet()), 'weights/baseline.json')
"
```

`weights/baseline.json` is already committed in this repo.

### 2 — Run the learner

```bash
python -m learner.run_learner --steps 500 --delta 2.0 --patience 50 --output weights/
```

All options:

| Flag | Default | Description |
|------|---------|-------------|
| `--steps` | 500 | Maximum optimization steps |
| `--delta` | 2.0 | Gaussian std-dev for weight perturbations |
| `--patience` | 50 | Non-improving steps before a random restart |
| `--output` | `weights/` | Directory for `best.json` and `log.jsonl` |
| `--seed` | 0 | Optimizer RNG seed (does not affect Copycat seeds) |
| `--weights` | baseline | Path to a starting weight JSON (resume from earlier run) |

### 3 — Apply learned weights to a new Copycat run

```python
import learner                           # must be imported first (applies shims)
from copycat.slipnet import Slipnet
from copycat.coderack import Coderack
from copycat.randomness import Randomness
from copycat.temperature import Temperature
from copycat.workspace import Workspace
from learner.weight_space import load, apply_weights
import contextlib, io

weights = load("weights/best.json")

class _Ctx: pass
ctx = _Ctx()
ctx.random = Randomness(seed=42)
ctx.slipnet = Slipnet()
ctx.temperature = Temperature()
ctx.workspace = Workspace(ctx)
ctx.coderack = Coderack(ctx)

apply_weights(ctx.slipnet, weights)
ctx.workspace.resetWithStrings("abc", "abd", "ijk")
with contextlib.redirect_stdout(io.StringIO()):
    ctx.temperature.useAdj("pbest")

ctx.coderack.reset(); ctx.slipnet.reset()
ctx.temperature.reset(); ctx.workspace.reset()

last_update = float("-inf")
while ctx.workspace.finalAnswer is None:
    t = ctx.coderack.codeletsRun
    ctx.temperature.tryUnclamp(t)
    if t >= last_update + 5:
        ctx.workspace.updateEverything()
        ctx.coderack.updateCodelets()
        ctx.slipnet.update(ctx.random)
        ctx.temperature.update(ctx.workspace.getUpdatedTemperature())
        last_update = t
    ctx.coderack.chooseAndRunCodelet()

print(ctx.workspace.finalAnswer, ctx.temperature.last_unclamped_value)
```

## Log format

`weights/log.jsonl` contains one JSON object per step:

```json
{
  "step": 42,
  "accepted": true,
  "restart": false,
  "perturbed_key": "letterCategory->samenessGroup",
  "old_temp": 23.45,
  "new_temp": 22.91,
  "best_temp": 22.91,
  "diagnostic": {
    "abc,abd,kji": 14.02,
    "abc,abd,xyz": 13.87
  }
}
```

| Field | Description |
|-------|-------------|
| `step` | Step index (0 = baseline evaluation) |
| `accepted` | Whether the candidate weights were adopted |
| `restart` | Whether this step was a forced random restart |
| `perturbed_key` | `"source->destination"` of the link that was changed |
| `old_temp` | Overall mean temp before this step |
| `new_temp` | Overall mean temp of the candidate weights |
| `best_temp` | Best overall mean temp seen so far |
| `diagnostic` | Per-diagnostic-problem mean temps for this candidate |

## Architecture

```
learner/
├── weight_space.py   extract / apply / perturb / serialize weight configs
├── evaluator.py      parallel evaluation: 45 Copycat runs per configuration
├── curriculum.py     problem set + diagnostic tracking
├── hill_climber.py   hill climbing loop with random restarts
└── run_learner.py    CLI entry point
```

The learner wraps Copycat without modifying any file inside `copycat/`.
Two thin shims are applied at import time (in `learner/__init__.py`):

1. **`inspect.getargspec`** — removed in Python 3.11; replaced with a
   4-field wrapper around `getfullargspec` to satisfy codeletMethods.py's
   assertion.
2. **GUI / matplotlib mocks** — injected into `sys.modules` so that the
   copycat package imports cleanly in headless environments without tkinter
   or matplotlib installed.
