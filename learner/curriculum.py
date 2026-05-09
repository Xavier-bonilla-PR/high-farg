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
    # ~98% of runs produce ijl. Clear unambiguous answer.
    ("abc", "abd", "ijk"):     {"ijl"},

    # Celebrated: kk treated as a GROUP → ll (~42% in this port).
    # ijkl (rightmost single letter k→l, ~51%) is the shallow answer.
    ("aabc", "aabd", "ijkk"):  {"ijll"},

    # kji is abc reversed; direction slippage → predecessor of i → kjh (~27%).
    # kjj (~50%) is sloppy; lji (~22%) is shallow. kjh has the lowest avg temp.
    ("abc", "abd", "kji"):     {"kjh"},

    # mrrkkk (~63%) is the most common and achievable answer in this port.
    # mrrjjjj (~4%) is the group-count-extension ideal but almost never seen.
    ("abc", "abd", "mrrjjj"):  {"mrrkkk", "mrrjjjj"},

    # rssuuu (~45%) is both celebrated (ttt GROUP → uuu) and commonly produced.
    # rssttu (~52%) is shallow (only the last t→u). rsstttt (group extension)
    # appears ~2% — rarely seen in this port.
    ("abc", "abd", "rssttt"):  {"rssuuu"},

    # xyd is what this port produces ~98% of the time.
    # wyz (celebrated: slip direction since z has no successor) is essentially
    # never produced by this port — included as the ideal aspirational target.
    ("abc", "abd", "xyz"):     {"xyd", "wyz"},

    # ijjlll (~46%): ttt GROUP → uuu, celebrated.
    # ijjkkl (~50%) is shallow (rightmost single letter k→l).
    # ijjkkkk (~2%, group extension) is rarely seen in this port.
    ("abc", "abd", "ijjkkk"):  {"ijjlll"},

    # xyu is produced ~100% of the time by this port and is the correct answer.
    # xyza (z wraps to a) is the theoretically elegant answer but not observed.
    ("rst", "rsu", "xyz"):     {"xyu"},

    # This port produces xyyd (~75%) and xyyzzd (~25%) — neither celebrated.
    # Celebrated answers (xyyzzzz group-extension, xyyaaaa wrap) are not
    # observed in this port. Include xyyd so the metric is not always 0%.
    ("abc", "abd", "xyyzzz"):  {"xyyd", "xyyzzzz", "xyyaaaa"},
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
