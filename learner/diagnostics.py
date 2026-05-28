"""
Diagnostic instrumentation for Slipnet weight analysis.

Runs Copycat trials while recording:
  - Per-node peak activation and full-activation events (node + codelet step)
  - Slippage events from workspace correspondences (which concept → which concept)
  - Total codelets run and slipnet update count
  - Whether the trial produced a preferred answer

Aggregates across multiple seeds to produce per-problem summaries:
  - activation_frequency : fraction of trials each node reached 100% activation
  - peak_activation      : mean peak activation per node across trials
  - slippage_counts      : {label: count} across all trials
  - slippage_pref_rate   : for each slippage label, % of those trials that also
                           produced a preferred answer (correlation proxy)
  - answer_distribution  : {answer: count}
  - preferred_rate        : fraction of trials with a preferred answer

Usage
-----
    from learner.diagnostics import run_diagnostic_pass
    results = run_diagnostic_pass(weight_dict, problems, preferred_answers_map, n_seeds=20)
"""

import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

DIAGNOSTIC_SEEDS = list(range(20))   # 20 seeds for richer stats


# ---------------------------------------------------------------------------
# Top-level worker — must be picklable (no closures)
# ---------------------------------------------------------------------------

def _run_diagnostic_trial(args):
    """Run one Copycat trial with full instrumentation.

    Returns a dict with:
        answer         : str
        temperature    : float
        codelets_run   : int
        slipnet_updates: int
        peak_activation: {node_name: float}   — max activation seen during the run
        full_activations: list of (node_name, codelet_step)
        slippages      : list of "initial->target" strings
    """
    weight_dict, initial, modified, target, seed = args

    _proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _proj_root not in sys.path:
        sys.path.insert(0, _proj_root)

    import inspect as _ins, collections as _col, types as _typ
    if not hasattr(_ins, "getargspec"):
        _AS = _col.namedtuple("ArgSpec", ["args", "varargs", "keywords", "defaults"])
        def _gas(f):
            s = _ins.getfullargspec(f)
            return _AS(s.args, s.varargs, s.varkw, s.defaults)
        _ins.getargspec = _gas

    def _stub(name):
        import types as _t
        m = _t.ModuleType(name)
        sys.modules[name] = m
        return m

    class _GUI:
        def __init__(self, *a, **kw): pass

    for _mn in ("copycat.gui", "copycat.gui.gui"):
        if _mn not in sys.modules:
            _m = _stub(_mn); _m.GUI = _GUI
    for _mn in ("matplotlib", "matplotlib.pyplot"):
        if _mn not in sys.modules:
            _m = _stub(_mn); _m.rcdefaults = lambda: None
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

    class _Context:
        pass

    ctx = _Context()
    ctx.random = Randomness(seed)
    ctx.slipnet = Slipnet()
    ctx.temperature = Temperature()
    ctx.workspace = Workspace(ctx)
    ctx.coderack = Coderack(ctx)

    apply_weights(ctx.slipnet, weight_dict)
    ctx.workspace.resetWithStrings(initial, modified, target)

    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        ctx.temperature.useAdj("pbest")

    ctx.coderack.reset()
    ctx.slipnet.reset()
    ctx.temperature.reset()
    ctx.workspace.reset()

    # Track per-node peak activations
    all_nodes = ctx.slipnet.slipnodes
    peak_activation = {n.name: 0.0 for n in all_nodes}
    full_activations = []   # (node_name, codelet_step)
    slipnet_updates = 0
    last_update = float("-inf")

    while ctx.workspace.finalAnswer is None:
        current_time = ctx.coderack.codeletsRun
        ctx.temperature.tryUnclamp(current_time)
        if current_time >= last_update + 5:
            ctx.workspace.updateEverything()
            ctx.coderack.updateCodelets()
            ctx.slipnet.update(ctx.random)
            slipnet_updates += 1

            # Snapshot activations right after each slipnet update
            for node in all_nodes:
                act = node.activation
                if act > peak_activation[node.name]:
                    peak_activation[node.name] = act
                if node.fully_active() and peak_activation[node.name] >= 99.9:
                    # Record first time this node reached full activation
                    entry = (node.name, current_time)
                    if not any(e[0] == node.name for e in full_activations):
                        full_activations.append(entry)

            ctx.temperature.update(ctx.workspace.getUpdatedTemperature())
            last_update = current_time
        ctx.coderack.chooseAndRunCodelet()

    # Collect slippages from workspace after completion
    slippages = []
    try:
        for mapping in ctx.workspace.slippages():
            initial_name = mapping.initialDescriptor.name
            target_name = mapping.targetDescriptor.name
            if initial_name != target_name:   # skip identity mappings
                slippages.append("{}->{}" .format(initial_name, target_name))
    except Exception:
        pass

    return {
        "answer": ctx.workspace.finalAnswer,
        "temperature": ctx.temperature.last_unclamped_value,
        "codelets_run": ctx.coderack.codeletsRun,
        "slipnet_updates": slipnet_updates,
        "peak_activation": peak_activation,
        "full_activations": full_activations,
        "slippages": slippages,
    }


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _aggregate_problem(trial_results, preferred_answers=None):
    """Aggregate list of trial dicts into a problem diagnostic summary."""
    n = len(trial_results)
    if n == 0:
        return {}

    temperatures  = [t["temperature"] for t in trial_results]
    answers       = [t["answer"] for t in trial_results]
    codelets_list = [t["codelets_run"] for t in trial_results]
    updates_list  = [t["slipnet_updates"] for t in trial_results]

    # ---- preferred labels ----
    pref_set = preferred_answers or set()
    is_preferred = [a in pref_set for a in answers]

    # ---- activation frequency & mean peak ----
    # union of all node names seen
    all_node_names = set()
    for t in trial_results:
        all_node_names.update(t["peak_activation"].keys())

    act_sum   = defaultdict(float)
    act_count = defaultdict(int)   # trials where node was seen
    full_freq = defaultdict(int)   # trials where node reached full activation

    for t in trial_results:
        for name, peak in t["peak_activation"].items():
            act_sum[name]   += peak
            act_count[name] += 1
            if peak >= 99.9:
                full_freq[name] += 1

    activation_frequency = {
        name: full_freq[name] / n for name in all_node_names
    }
    mean_peak_activation = {
        name: act_sum[name] / act_count[name] if act_count[name] else 0.0
        for name in all_node_names
    }

    # ---- slippage aggregation ----
    slippage_counts = defaultdict(int)
    # Map from slippage label → list of booleans (did that trial produce preferred answer?)
    slippage_pref = defaultdict(list)

    for t, pref in zip(trial_results, is_preferred):
        seen_in_trial = set()
        for s in t["slippages"]:
            slippage_counts[s] += 1
            if s not in seen_in_trial:
                slippage_pref[s].append(pref)
                seen_in_trial.add(s)

    slippage_pref_rate = {
        s: sum(vals) / len(vals) if vals else 0.0
        for s, vals in slippage_pref.items()
    }

    # ---- answer distribution ----
    answer_dist = defaultdict(int)
    for a in answers:
        answer_dist[a] += 1

    return {
        "n_trials": n,
        "mean_temperature": sum(temperatures) / n,
        "preferred_rate": sum(is_preferred) / n,
        "mean_codelets": sum(codelets_list) / n,
        "mean_slipnet_updates": sum(updates_list) / n,
        "answer_distribution": dict(answer_dist),
        # Keep top-30 nodes by activation frequency to keep output manageable
        "activation_frequency": dict(
            sorted(activation_frequency.items(), key=lambda x: -x[1])[:30]
        ),
        "mean_peak_activation": dict(
            sorted(mean_peak_activation.items(), key=lambda x: -x[1])[:30]
        ),
        "slippage_counts": dict(
            sorted(slippage_counts.items(), key=lambda x: -x[1])
        ),
        "slippage_pref_rate": slippage_pref_rate,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_diagnostic_pass(weight_dict, problems, preferred_answers_map=None, n_seeds=20):
    """Run diagnostic trials for all problems; return per-problem summaries.

    Parameters
    ----------
    weight_dict : dict
    problems : list of (initial, modified, target) tuples
    preferred_answers_map : dict or None  — {problem_tuple: set_of_preferred_answers}
    n_seeds : int — number of seeds per problem (default 20)

    Returns
    -------
    dict  {problem_tuple: diagnostic_summary_dict}
    """
    seeds = list(range(n_seeds))
    all_args = [
        (weight_dict, p[0], p[1], p[2], s)
        for p in problems
        for s in seeds
    ]
    with ProcessPoolExecutor() as executor:
        all_results = list(executor.map(_run_diagnostic_trial, all_args))

    pref_map = preferred_answers_map or {}
    return {
        problem: _aggregate_problem(
            all_results[i * n_seeds: (i + 1) * n_seeds],
            preferred_answers=pref_map.get(problem),
        )
        for i, problem in enumerate(problems)
    }
