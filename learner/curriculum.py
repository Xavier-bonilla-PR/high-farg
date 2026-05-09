"""
Problem set management and diagnostic temperature tracking.

Loads all 9 problems from input/problems.csv and tags two of them as
diagnostic:
  • Problem 3  (abc, abd, kji)  — reverse-alphabet target, tests direction slip
  • Problem 6  (abc, abd, xyz)  — late-alphabet successor target

Diagnostic problems appear individually in every log entry so we can watch
how the optimizer affects the hardest / most interesting cases.
"""

import os

# 0-based indices of the two diagnostic problems in problems.csv
DIAGNOSTIC_INDICES = (2, 5)


def load_problems(path=None):
    """Return list of (initial, modified, target) tuples from problems.csv."""
    if path is None:
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "input", "problems.csv",
        )
    problems = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) == 3:
                problems.append(tuple(parts))
    return problems


class Curriculum:
    """Manages the problem set and aggregates per-step diagnostic statistics."""

    def __init__(self, problems_path=None):
        self.problems = load_problems(problems_path)
        self.diagnostic_problems = [
            self.problems[i] for i in DIAGNOSTIC_INDICES
        ]

    # ------------------------------------------------------------------

    def evaluate_all(self, weight_dict, evaluator_fn=None):
        """Evaluate all problems and return aggregated results.

        Parameters
        ----------
        weight_dict : dict
            Current weight configuration.
        evaluator_fn : callable or None
            Optional ``(weight_dict, problem) -> stats_dict`` function.
            When None (default) the batched parallel evaluator is used,
            which is far more efficient for full-curriculum evaluation.

        Returns
        -------
        problem_stats : dict
            ``{problem_tuple: stats_dict}`` — per-problem mean, variance,
            failures, and answer distribution.
        overall_stats : dict
            Aggregated stats across all problems and all trials.
        diagnostic_stats : dict
            ``{problem_tuple: stats_dict}`` for the two diagnostic problems.
        """
        from learner.evaluator import evaluate_problems_batched
        from learner.monitor import compute_overall_stats

        if evaluator_fn is None:
            problem_stats = evaluate_problems_batched(weight_dict, self.problems)
        else:
            problem_stats = {p: evaluator_fn(weight_dict, p) for p in self.problems}

        overall_stats = compute_overall_stats(problem_stats)
        diagnostic_stats = {p: problem_stats[p] for p in self.diagnostic_problems}

        return problem_stats, overall_stats, diagnostic_stats

    def problem_label(self, problem):
        """Human-readable label for a problem tuple, e.g. 'abc→abd/kji'."""
        initial, modified, target = problem
        return "{}→{}/{}".format(initial, modified, target)
