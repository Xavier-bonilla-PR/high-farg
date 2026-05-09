"""
Entry point for the Slipnet weight learner.

Usage
-----
    python -m learner.run_learner [options]

Options
-------
--steps    INT    Maximum optimisation steps          (default: 500)
--delta    FLOAT  Gaussian perturbation std-dev       (default: 2.0)
--patience INT    Non-improving steps before restart  (default: 50)
--output   DIR    Output directory for weights/log    (default: weights/)
--seed     INT    RNG seed for the optimiser itself   (default: 0)
--weights  FILE   Start from an existing weight file  (default: baseline)

Output files
------------
weights/best.json   Best weights found, updated after each improvement.
weights/log.jsonl   Full step log (one JSON object per line).
"""

import argparse
import os
import sys

# Make sure the project root is importable regardless of how this is invoked.
_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)


def _build_baseline(output_dir):
    """Extract and save the original Mitchell weights as baseline.json."""
    from copycat.slipnet import Slipnet
    from learner.weight_space import extract_weights, save

    sn = Slipnet()
    weights = extract_weights(sn)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "baseline.json")
    save(weights, path)
    print("Baseline weights written to {}".format(path))
    return weights


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Hill-climb Slipnet link weights to minimise mean Copycat temperature."
    )
    parser.add_argument("--steps", type=int, default=500, help="Max optimisation steps")
    parser.add_argument("--delta", type=float, default=2.0, help="Perturbation std-dev")
    parser.add_argument("--patience", type=int, default=50, help="Steps before restart")
    parser.add_argument("--output", default="weights", help="Output directory")
    parser.add_argument("--seed", type=int, default=0, help="Optimiser RNG seed")
    parser.add_argument(
        "--weights",
        default=None,
        help="Path to initial weight JSON (default: generate baseline)",
    )
    args = parser.parse_args(argv)

    # ------------------------------------------------------------------ #
    # 1. Initialise weights                                                #
    # ------------------------------------------------------------------ #
    if args.weights is not None:
        from learner.weight_space import load
        initial_weights = load(args.weights)
        print("Loaded initial weights from {}".format(args.weights))
    else:
        initial_weights = _build_baseline(args.output)

    # ------------------------------------------------------------------ #
    # 2. Set up curriculum                                                 #
    # ------------------------------------------------------------------ #
    from learner.curriculum import Curriculum
    curriculum = Curriculum()

    print("\nProblems:")
    for i, p in enumerate(curriculum.problems, 1):
        tag = " ← diagnostic" if p in curriculum.diagnostic_problems else ""
        print("  {:2d}. {},{},{}{}".format(i, p[0], p[1], p[2], tag))
    print()

    # ------------------------------------------------------------------ #
    # 3. Run the optimiser                                                 #
    # ------------------------------------------------------------------ #
    from learner.hill_climber import HillClimber
    climber = HillClimber(
        initial_weights=initial_weights,
        curriculum=curriculum,
        output_dir=args.output,
        delta=args.delta,
        patience=args.patience,
        max_steps=args.steps,
        rng_seed=args.seed,
    )
    best_weights, best_temp = climber.run()

    print("\nBest weights saved to {}".format(os.path.join(args.output, "best.json")))
    print("Step log saved to {}".format(os.path.join(args.output, "log.jsonl")))
    print("Best overall mean temperature: {:.4f}".format(best_temp))


if __name__ == "__main__":
    main()
