"""
Hill-climbing optimizer for Slipnet link weights.

Algorithm
---------
1. Start from the baseline (original Mitchell) weights.
2. At each step, perturb one randomly chosen non-zero link weight by N(0, delta).
3. Evaluate all 9 problems with the candidate weights.
4. Accept the candidate if overall_mean < current_overall_mean; else revert.
5. After `patience` consecutive non-improving steps, trigger a random restart:
   reset to best-known weights and apply a larger perturbation (delta × 5)
   to escape the local minimum.
6. Stop after `max_steps` total steps or when no improvement has occurred
   in the last 100 steps.

Key parameters
--------------
delta    : std-dev of the Gaussian perturbation (default 2.0 — conservative
           because link lengths are in [0, 100] and the system is sensitive).
patience : steps without improvement before a restart (default 50).
max_steps: hard limit on total optimization steps (default 500).
"""

import json
import os
import random

from learner.curriculum import Curriculum
from learner.monitor import Monitor
from learner.weight_space import perturb, save


NO_IMPROVE_WINDOW = 100  # convergence check window


class HillClimber:
    """Standard hill climber with random restarts for Slipnet weight learning."""

    def __init__(
        self,
        initial_weights,
        curriculum,
        output_dir="weights",
        delta=2.0,
        patience=50,
        max_steps=500,
        rng_seed=0,
        answer_weight=10.0,
    ):
        self.curriculum = curriculum
        self.output_dir = output_dir
        self.delta = delta
        self.patience = patience
        self.max_steps = max_steps
        self.answer_weight = answer_weight
        self.rng = random.Random(rng_seed)

        os.makedirs(output_dir, exist_ok=True)
        self.monitor = Monitor(output_dir)

        self.current_weights = dict(initial_weights)
        self.best_weights = dict(initial_weights)
        self.best_overall = float("inf")
        self.current_overall = float("inf")

        self.step_log = []       # full history
        self.no_improve_count = 0

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------

    def _objective(self, overall_stats):
        """Combined objective = mean_temp + answer_weight * wrong_answer_fraction.

        answer_weight (default 10.0) is in temperature units.  A value of 10
        means 100% wrong answers adds 10 degrees to the effective temperature.
        Set answer_weight=0 to optimise purely on temperature.
        """
        mean = overall_stats["mean"]
        if self.answer_weight == 0 or "preferred_rate" not in overall_stats:
            return mean
        wrong_rate = 1.0 - overall_stats["preferred_rate"]
        return mean + self.answer_weight * wrong_rate

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self):
        """Run the hill-climbing loop; return best_weights and best_overall_temp."""
        print("Initialising: evaluating baseline weights …")
        prob_stats, overall_stats, diag_stats = self.curriculum.evaluate_all(
            self.current_weights
        )
        self.current_overall = self._objective(overall_stats)
        self.best_overall = self.current_overall
        self.best_weights = dict(self.current_weights)
        self._log_step(
            step=0, accepted=True, perturbed_key=None, restart=False,
            old_temp=self.current_overall, new_temp=self.current_overall,
            effective_delta=self.delta,
            overall_stats=overall_stats, diagnostic_stats=diag_stats,
        )
        self.monitor.record(
            step=0, accepted=True, restart=False,
            overall_stats=overall_stats, problem_stats=prob_stats,
            diagnostic_stats=diag_stats, curriculum=self.curriculum,
        )
        print(
            "  Baseline objective: {:.4f}  (mean={:.4f}  pref={:.0f}%)".format(
                self.current_overall,
                overall_stats["mean"],
                overall_stats.get("preferred_rate", float("nan")) * 100,
            )
        )
        self._save_best()

        steps_since_any_improve = 0

        for step in range(1, self.max_steps + 1):
            restart = False

            # --- random restart after patience exhausted ---
            if self.no_improve_count >= self.patience:
                # Jump unconditionally to a large perturbation of best weights.
                candidate_weights, key = perturb(
                    self.best_weights, self.delta * 5, self.rng
                )
                self.no_improve_count = 0
                restart = True
                effective_delta = self.delta  # reset for logging
            else:
                # Adaptive delta: decay linearly from delta → delta*0.1 as
                # no_improve_count rises toward patience.  Smaller moves when
                # we are already close to giving up on this basin, larger moves
                # after a fresh start or recent acceptance.
                decay = 1.0 - 0.9 * (self.no_improve_count / self.patience)
                effective_delta = self.delta * decay
                candidate_weights, key = perturb(
                    self.current_weights, effective_delta, self.rng
                )

            prob_stats, overall_stats, diag_stats = self.curriculum.evaluate_all(
                candidate_weights
            )
            new_overall = self._objective(overall_stats)
            old_temp = self.current_overall

            if restart:
                accepted = True
                self.current_weights = candidate_weights
                self.current_overall = new_overall
            else:
                accepted = new_overall < self.current_overall
                if accepted:
                    self.current_weights = candidate_weights
                    self.current_overall = new_overall
                    self.no_improve_count = 0
                    steps_since_any_improve = 0
                else:
                    self.no_improve_count += 1
                    steps_since_any_improve += 1

            if new_overall < self.best_overall:
                self.best_overall = new_overall
                self.best_weights = dict(candidate_weights)
                self._save_best()

            key_label = (
                "{}->{}".format(key[0], key[1]) if key is not None else "—"
            )
            self._print_step(
                step, self.current_overall, self.best_overall,
                key_label, accepted, effective_delta, diag_stats,
            )
            self._log_step(
                step=step, accepted=accepted, perturbed_key=key_label,
                restart=restart, old_temp=old_temp, new_temp=new_overall,
                effective_delta=effective_delta,
                overall_stats=overall_stats, diagnostic_stats=diag_stats,
            )
            self.monitor.record(
                step=step, accepted=accepted, restart=restart,
                overall_stats=overall_stats, problem_stats=prob_stats,
                diagnostic_stats=diag_stats, curriculum=self.curriculum,
            )
            self._flush_log()

            # convergence check
            if steps_since_any_improve >= NO_IMPROVE_WINDOW:
                print(
                    "\nConverged: no improvement in last {} steps.".format(
                        NO_IMPROVE_WINDOW
                    )
                )
                break

        print("\nDone.  Best overall mean temp: {:.4f}".format(self.best_overall))
        return self.best_weights, self.best_overall

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_best(self):
        path = os.path.join(self.output_dir, "best.json")
        save(self.best_weights, path)

    def _print_step(self, step, current, best, key_label, accepted,
                    effective_delta, diag_stats):
        diag_parts = "  ".join(
            "{}: {:.2f}".format(self.curriculum.problem_label(p), s["mean"])
            for p, s in diag_stats.items()
        )
        status = "✓ accepted" if accepted else "✗ rejected"
        print(
            "step {:4d} | cur {:.4f} | best {:.4f} | δ={:.3f} | {} | {}  [diag: {}]".format(
                step, current, best, effective_delta, status, key_label, diag_parts
            )
        )

    def _log_step(self, step, accepted, perturbed_key, restart,
                  old_temp, new_temp, effective_delta, overall_stats, diagnostic_stats):
        from learner.monitor import _serialisable
        entry = {
            "step": step,
            "accepted": accepted,
            "restart": restart,
            "perturbed_key": perturbed_key,
            "effective_delta": round(effective_delta, 4),
            "old_temp": old_temp,
            "new_temp": new_temp,
            "best_temp": self.best_overall,
            "overall": _serialisable(overall_stats),
            "diagnostic": {
                "{},{},{}".format(*p): _serialisable(s)
                for p, s in diagnostic_stats.items()
            },
        }
        self.step_log.append(entry)

    def _flush_log(self):
        """Append the last log entry to the JSONL file."""
        log_path = os.path.join(self.output_dir, "log.jsonl")
        with open(log_path, "a") as fh:
            fh.write(json.dumps(self.step_log[-1]) + "\n")
