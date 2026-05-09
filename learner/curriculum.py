"""
Problem set management and diagnostic temperature tracking.

Loads all 9 problems from input/problems.csv and tags two of them as
diagnostic:
  • Problem 3  (abc, abd, kji)  — reverse-alphabet target, tests direction slip
  • Problem 6  (abc, abd, xyz)  — late-alphabet successor target

Diagnostic problems appear individually in every log entry so we can watch
how the optimizer affects the hardest / most interesting cases.

PREFERRED_ANSWERS defines the celebrated / ideal answer(s) for each problem
drawn from Hofstadter & Mitchell "Fluid Concepts and Creative Analogies" (1995)
and Mitchell's 1993 thesis.  These are used to compute preferred_rate — the
fraction of trials that produce an insightful answer — which the hill climber
can include in its objective alongside mean temperature.
"""

import os

# 0-based indices of the two diagnostic problems in problems.csv
DIAGNOSTIC_INDICES = (2, 5)

# Preferred (celebrated / insightful) answers per problem.
# Multiple entries mean any of those answers is considered ideal.
# Sources: Hofstadter & Mitchell "Fluid Concepts and Creative Analogies" 1995,
#          Mitchell 1993 thesis, fargonauts/copycat-java result distributions.
#
# The key insight for problems 4, 5, 7, 9: perceiving the target as LENGTH-
# BASED groups (mrrjjj = 1+2+3 letters) and applying "successor" to the GROUP
# COUNT (add one more letter to the last group) rather than to the letter
# itself.  This group-extension answer is more celebrated than the letter-
# successor answer (e.g. mrrjjjj is preferred over mrrkkk).
PREFERRED_ANSWERS = {
    # Straightforward: c→d maps to k→l.
    ("abc", "abd", "ijk"):     {"ijl"},

    # Rightmost GROUP kk → ll (group successor, not single letter).
    ("aabc", "aabd", "ijkk"):  {"ijll"},

    # kji is abc reversed; direction slippage → predecessor of i → kjh.
    # kjh is the most frequent answer AND the most celebrated.
    ("abc", "abd", "kji"):     {"kjh"},

    # Group extension: jjj (3 j's) → jjjj (4 j's) is the insightful answer.
    # mrrkkk (letter successor) is the shallow answer.
    ("abc", "abd", "mrrjjj"):  {"mrrjjjj", "mrrkkk"},

    # Group extension: ttt (3 t's) → tttt (4 t's).
    # rssuuu (letter successor) is the shallow answer.
    ("abc", "abd", "rssttt"):  {"rsstttt", "rssuuu"},

    # z has no successor; celebrated answer slips direction → wyz.
    # xyd (z slips to d) is the common shallow answer.
    ("abc", "abd", "xyz"):     {"wyz", "xyd"},

    # Group extension: kkk (3 k's) → kkkk (4 k's) is the insightful answer.
    # ijjlll (letter successor) is the shallow answer.
    ("abc", "abd", "ijjkkk"):  {"ijjkkkk", "ijjlll"},

    # t→u maps z→z's successor = a (alphabet wraps) → xyza.
    # xyu (literal u substitution slip) is also common.
    ("rst", "rsu", "xyz"):     {"xyza", "xyu"},

    # Group extension: zzz (3 z's) → zzzz (4 z's) is the insightful answer.
    # xyyaaa (z wraps to a) is the alphabet-wrap answer.
    ("abc", "abd", "xyyzzz"):  {"xyyzzzz", "xyyaaa", "xyyaaaa"},
}


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
            failures, preferred_rate, and answer distribution.
        overall_stats : dict
            Aggregated stats across all problems and all trials.
        diagnostic_stats : dict
            ``{problem_tuple: stats_dict}`` for the two diagnostic problems.
        """
        from learner.evaluator import evaluate_problems_batched
        from learner.monitor import compute_overall_stats

        if evaluator_fn is None:
            problem_stats = evaluate_problems_batched(
                weight_dict, self.problems, PREFERRED_ANSWERS
            )
        else:
            problem_stats = {p: evaluator_fn(weight_dict, p) for p in self.problems}

        overall_stats = compute_overall_stats(problem_stats)
        diagnostic_stats = {p: problem_stats[p] for p in self.diagnostic_problems}

        return problem_stats, overall_stats, diagnostic_stats

    def problem_label(self, problem):
        """Human-readable label for a problem tuple, e.g. 'abc→abd/kji'."""
        initial, modified, target = problem
        return "{}→{}/{}".format(initial, modified, target)
