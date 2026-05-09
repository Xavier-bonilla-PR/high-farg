"""
Per-step statistics monitoring for the hill-climbing run.

Tracks four metrics across all 45 trials (9 problems × 5 seeds) per step:

  mean                — mean final temperature (the optimisation target)
  variance / std      — spread of temperatures; high variance = unpredictable
  failures            — trials that finished with temp > FAILURE_THRESHOLD (60)
  answer_distribution — how often each output string was produced

Results are written to monitor.jsonl (one JSON object per step) and printed
as a compact summary line after each step.
"""

import json
import math
import os

FAILURE_THRESHOLD = 60.0


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def compute_stats(trial_results):
    """Return a stats dict from a list of (temp, answer) tuples for one problem."""
    temps = [t for t, _ in trial_results]
    answers = [a for _, a in trial_results]

    n = len(temps)
    mean = sum(temps) / n
    variance = sum((t - mean) ** 2 for t in temps) / n
    failures = sum(1 for t in temps if t > FAILURE_THRESHOLD)

    answer_dist = {}
    for a in answers:
        answer_dist[a] = answer_dist.get(a, 0) + 1
    # Sort by frequency descending for readability
    answer_dist = dict(sorted(answer_dist.items(), key=lambda kv: -kv[1]))

    return {
        "n_trials": n,
        "mean": mean,
        "variance": variance,
        "std": math.sqrt(variance),
        "failures": failures,
        "failure_rate": failures / n,
        "answer_distribution": answer_dist,
    }


def compute_overall_stats(problem_stats):
    """Aggregate stats across all problems into a single overall stats dict.

    Pools all individual trial results so that variance and failure counts
    reflect the full 45-trial distribution, not just averages of averages.
    """
    all_means = [s["mean"] for s in problem_stats.values()]
    all_variances = [s["variance"] for s in problem_stats.values()]
    all_failures = [s["failures"] for s in problem_stats.values()]
    all_n = sum(s["n_trials"] for s in problem_stats.values())

    # Combined answer distribution across all problems
    combined_answers = {}
    for s in problem_stats.values():
        for ans, cnt in s["answer_distribution"].items():
            combined_answers[ans] = combined_answers.get(ans, 0) + cnt
    combined_answers = dict(sorted(combined_answers.items(), key=lambda kv: -kv[1]))

    # Mean of problem means (equal weight per problem)
    overall_mean = sum(all_means) / len(all_means)

    # Variance: average within-problem variance (law of total variance, ignoring
    # between-problem term since problems are not commensurable).
    overall_variance = sum(all_variances) / len(all_variances)

    total_failures = sum(all_failures)

    return {
        "n_trials": all_n,
        "mean": overall_mean,
        "variance": overall_variance,
        "std": math.sqrt(overall_variance),
        "failures": total_failures,
        "failure_rate": total_failures / all_n,
        "answer_distribution": combined_answers,
    }


# ---------------------------------------------------------------------------
# Monitor class
# ---------------------------------------------------------------------------

class Monitor:
    """Writes per-step monitoring data to monitor.jsonl and prints summaries."""

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.path = os.path.join(output_dir, "monitor.jsonl")
        os.makedirs(output_dir, exist_ok=True)

    def record(self, step, accepted, restart, overall_stats, problem_stats,
               diagnostic_stats, curriculum):
        """Append one monitoring entry to monitor.jsonl and print a summary line."""
        entry = {
            "step": step,
            "accepted": accepted,
            "restart": restart,
            "overall": _serialisable(overall_stats),
            "diagnostic": {
                "{},{},{}".format(*p): _serialisable(s)
                for p, s in diagnostic_stats.items()
            },
            "per_problem": {
                "{},{},{}".format(*p): _serialisable(s)
                for p, s in problem_stats.items()
            },
        }
        with open(self.path, "a") as fh:
            fh.write(json.dumps(entry) + "\n")

        self._print_summary(overall_stats, diagnostic_stats, curriculum)

    def _print_summary(self, overall, diagnostic, curriculum):
        top_answers = ", ".join(
            "{}×{}".format(cnt, ans)
            for ans, cnt in list(overall["answer_distribution"].items())[:4]
        )
        diag_parts = "  ".join(
            "{} fail={} σ={:.1f}".format(
                curriculum.problem_label(p),
                s["failures"],
                s["std"],
            )
            for p, s in diagnostic.items()
        )
        print(
            "        σ={:.2f}  fail={}/{} ({:.0f}%)  answers=[{}]  diag: {}".format(
                overall["std"],
                overall["failures"],
                overall["n_trials"],
                overall["failure_rate"] * 100,
                top_answers,
                diag_parts,
            )
        )


def _serialisable(stats):
    """Return a copy of a stats dict safe for json.dumps (round floats)."""
    out = {}
    for k, v in stats.items():
        if isinstance(v, float):
            out[k] = round(v, 6)
        else:
            out[k] = v
    return out
