"""
Run a diagnostic pass on the best-known weights and print a report.

Usage
-----
    python -m learner.run_diagnostics [options]

Options
-------
--weights FILE   Weight file to diagnose (default: weights/best.json)
--seeds  INT     Number of random seeds per problem (default: 20)
--output FILE    Save diagnostics JSON to this file
                 (default: weights/diagnostics.json)
--top    INT     Number of top activation nodes / slippages to show (default: 10)

Output
------
Prints a human-readable table per problem showing:
  - Temperature, preferred-answer rate
  - Top-N most activated slipnodes (by activation frequency)
  - Top-N slippage events (and their correlation with preferred answers)

Also writes full data to JSON for downstream analysis.
"""

import argparse
import json
import os
import sys

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)


def _fmt_pct(v):
    return "{:.0f}%".format(v * 100)


def _print_report(results, curriculum, top_n):
    pref_map = __import__("learner.curriculum", fromlist=["PREFERRED_ANSWERS"]).PREFERRED_ANSWERS
    sep = "─" * 72

    for problem, diag in results.items():
        label = curriculum.problem_label(problem)
        pref_answers = pref_map.get(problem, set())
        print("\n" + sep)
        print("Problem: {}   preferred: {}".format(label, ", ".join(sorted(pref_answers)) or "—"))
        print("  mean_temp={:.2f}  pref_rate={}  n_trials={}  "
              "mean_codelets={:.0f}  slipnet_updates={:.0f}".format(
            diag["mean_temperature"],
            _fmt_pct(diag["preferred_rate"]),
            diag["n_trials"],
            diag["mean_codelets"],
            diag["mean_slipnet_updates"],
        ))

        print("\n  Answer distribution:")
        ans_sorted = sorted(diag["answer_distribution"].items(), key=lambda x: -x[1])
        n = diag["n_trials"]
        for ans, cnt in ans_sorted[:8]:
            bar = "█" * int(cnt / n * 30)
            print("    {:12s} {:3d}/{:<3d} ({:4.0f}%)  {}".format(
                ans, cnt, n, cnt / n * 100, bar))

        print("\n  Top-{} slipnodes by full-activation frequency:".format(top_n))
        act_freq = diag.get("activation_frequency", {})
        for name, freq in list(act_freq.items())[:top_n]:
            bar = "█" * int(freq * 20)
            print("    {:25s} {}  ({})".format(name, bar, _fmt_pct(freq)))

        print("\n  Top-{} slippage events (count / pref-rate when this slippage occurs):".format(top_n))
        slip_counts = diag.get("slippage_counts", {})
        slip_pref   = diag.get("slippage_pref_rate", {})
        for slip, cnt in list(slip_counts.items())[:top_n]:
            prate = slip_pref.get(slip, 0.0)
            bar = "█" * int(prate * 20)
            print("    {:30s}  {:3d}x  pref_rate={}  {}".format(
                slip, cnt, _fmt_pct(prate), bar))

    print("\n" + sep)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default="weights/best.json",
                        help="Weight file to diagnose (default: weights/best.json)")
    parser.add_argument("--seeds", type=int, default=20,
                        help="Seeds per problem (default: 20)")
    parser.add_argument("--output", default="weights/diagnostics.json",
                        help="Output JSON path (default: weights/diagnostics.json)")
    parser.add_argument("--top", type=int, default=10,
                        help="Top N items to show in report (default: 10)")
    args = parser.parse_args(argv)

    from learner.weight_space import load
    from learner.curriculum import Curriculum, PREFERRED_ANSWERS
    from learner.diagnostics import run_diagnostic_pass

    weights = load(args.weights)
    curriculum = Curriculum()

    print("Loaded weights from {}".format(args.weights))
    print("Running {} seeds × {} problems = {} trials …".format(
        args.seeds, len(curriculum.problems), args.seeds * len(curriculum.problems)))

    results = run_diagnostic_pass(
        weights, curriculum.problems, PREFERRED_ANSWERS, n_seeds=args.seeds
    )

    _print_report(results, curriculum, top_n=args.top)

    # Serialise — convert tuple keys to strings
    out = {
        "{},{},{}".format(*p): diag for p, diag in results.items()
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nFull diagnostics saved to {}".format(args.output))


if __name__ == "__main__":
    main()
