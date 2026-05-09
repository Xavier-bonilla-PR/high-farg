"""
Parallel evaluator: maps a weight configuration to a mean final workspace
temperature for a given Copycat problem.

Each evaluation runs Copycat once per seed from SEEDS using a fresh Slipnet
with the supplied weights applied.  All seed-runs for a set of problems are
dispatched together into a single ProcessPoolExecutor so the OS can schedule
them across available cores.

Worker function (_run_single) is a module-level function — not a closure —
so it can be pickled by multiprocessing on any platform.
"""

import inspect as _inspect
import os
import sys
from concurrent.futures import ProcessPoolExecutor

# inspect.getargspec was removed in Python 3.11; patch before copycat imports.
# Must return a 4-field namedtuple matching the old ArgSpec shape.
if not hasattr(_inspect, "getargspec"):
    import collections as _coll
    _ArgSpec = _coll.namedtuple("ArgSpec", ["args", "varargs", "keywords", "defaults"])

    def _getargspec(f):
        s = _inspect.getfullargspec(f)
        return _ArgSpec(s.args, s.varargs, s.varkw, s.defaults)

    _inspect.getargspec = _getargspec

SEEDS = [42, 7, 13, 99, 271]


# ---------------------------------------------------------------------------
# Top-level worker (must be importable by subprocesses — no closures).
# ---------------------------------------------------------------------------

def _run_single(args):
    """Run one Copycat trial with the given weights and seed; return final temperature.

    Builds the Copycat context from individual components to avoid importing
    copycat.copycat (which transitively imports the GUI / tkinter).  The
    logic mirrors Copycat.runTrial() exactly.
    """
    weight_dict, initial, modified, target, seed = args

    # Ensure the project root is on sys.path so both 'copycat' and 'learner'
    # are importable when the worker is spawned in a fresh process.
    _proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _proj_root not in sys.path:
        sys.path.insert(0, _proj_root)

    # inspect.getargspec compatibility shim (removed in Python 3.11).
    import inspect as _ins, collections as _col, types as _typ
    if not hasattr(_ins, "getargspec"):
        _AS = _col.namedtuple("ArgSpec", ["args", "varargs", "keywords", "defaults"])
        def _gas(f):
            s = _ins.getfullargspec(f)
            return _AS(s.args, s.varargs, s.varkw, s.defaults)
        _ins.getargspec = _gas

    # Stub out headless/missing libraries so copycat.__init__ can import.
    def _stub(name):
        m = _typ.ModuleType(name)
        sys.modules[name] = m
        return m

    class _GUI:
        def __init__(self, *a, **kw): pass

    for _mn in ("copycat.gui", "copycat.gui.gui"):
        if _mn not in sys.modules:
            _m = _stub(_mn); _m.GUI = _GUI

    for _mn in ("matplotlib", "matplotlib.pyplot"):
        if _mn not in sys.modules:
            _m = _stub(_mn)
            _m.rcdefaults = lambda: None

    if "numpy" not in sys.modules:
        _stub("numpy")

    if "copycat.plot" not in sys.modules:
        _m = _stub("copycat.plot"); _m.plot_answers = lambda *a, **kw: None

    from copycat.coderack import Coderack
    from copycat.randomness import Randomness
    from copycat.slipnet import Slipnet
    from copycat.temperature import Temperature
    from copycat.workspace import Workspace
    from learner.weight_space import apply_weights

    # Minimal context object that satisfies all component cross-references.
    class _Context:
        pass

    ctx = _Context()
    ctx.random = Randomness(seed)
    ctx.slipnet = Slipnet()
    ctx.temperature = Temperature()
    ctx.workspace = Workspace(ctx)
    ctx.coderack = Coderack(ctx)

    # Apply learned weights before any reset so they persist through
    # slipnet.reset() (which only touches node activations, not link lengths).
    apply_weights(ctx.slipnet, weight_dict)

    ctx.workspace.resetWithStrings(initial, modified, target)

    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        ctx.temperature.useAdj("pbest")

    # Mirror Copycat.runTrial() reset sequence.
    ctx.coderack.reset()
    ctx.slipnet.reset()
    ctx.temperature.reset()
    ctx.workspace.reset()

    last_update = float("-inf")
    while ctx.workspace.finalAnswer is None:
        current_time = ctx.coderack.codeletsRun
        ctx.temperature.tryUnclamp(current_time)
        if current_time >= last_update + 5:
            ctx.workspace.updateEverything()
            ctx.coderack.updateCodelets()
            ctx.slipnet.update(ctx.random)
            ctx.temperature.update(ctx.workspace.getUpdatedTemperature())
            last_update = current_time
        ctx.coderack.chooseAndRunCodelet()

    return ctx.temperature.last_unclamped_value, ctx.workspace.finalAnswer


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_problem(weight_dict, problem):
    """Return stats dict for a single problem across SEEDS.

    Opens a fresh ProcessPoolExecutor for the seed-runs.  Prefer
    evaluate_problems_batched() when evaluating multiple problems to amortise
    pool startup cost.
    """
    from learner.monitor import compute_stats
    initial, modified, target = problem
    args_list = [
        (weight_dict, initial, modified, target, s) for s in SEEDS
    ]
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(_run_single, args_list))
    return compute_stats(results)


def evaluate_problems_batched(weight_dict, problems, preferred_answers_map=None):
    """Return ``{problem: stats_dict}`` for all problems using one shared pool.

    All len(problems) × len(SEEDS) workers are submitted together so the OS
    scheduler can keep all cores busy.  Each stats dict contains mean,
    variance, std, failures, failure_rate, answer_distribution, and —
    when preferred_answers_map is provided — preferred_rate.

    Parameters
    ----------
    preferred_answers_map : dict or None
        ``{problem_tuple: set_of_preferred_answer_strings}``
        Typically ``curriculum.PREFERRED_ANSWERS``.
    """
    from learner.monitor import compute_stats
    all_args = [
        (weight_dict, p[0], p[1], p[2], s)
        for p in problems
        for s in SEEDS
    ]
    with ProcessPoolExecutor() as executor:
        all_results = list(executor.map(_run_single, all_args))

    n = len(SEEDS)
    return {
        problem: compute_stats(
            all_results[i * n: (i + 1) * n],
            preferred_answers=(preferred_answers_map or {}).get(problem),
        )
        for i, problem in enumerate(problems)
    }
