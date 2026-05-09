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
    ):
        self.curriculum = curriculum
        self.output_dir = output_dir
        self.delta = delta
        self.patience = patience
        self.max_steps = max_steps
        self.rng = random.Random(rng_seed)

        os.makedirs(output_dir, exist_ok=True)

        self.current_weights = dict(initial_weights)
        self.best_weights = dict(initial_weights)
        self.best_overall = float("inf")
        self.current_overall = float("inf")

        self.step_log = []       # full history
        self.no_improve_count = 0

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self):
        """Run the hill-climbing loop; return best_weights and best_overall_temp."""
        print("Initialising: evaluating baseline weights …")
        _, self.current_overall, diag = self.curriculum.evaluate_all(
            self.current_weights
        )
        self.best_overall = self.current_overall
        self.best_weights = dict(self.current_weights)
        self._log_step(
            step=0,
            accepted=True,
            perturbed_key=None,
            old_temp=self.current_overall,
            new_temp=self.current_overall,
            diagnostic_temps=diag,
            restart=False,
        )
        print(
            "  Baseline overall mean temp: {:.4f}".format(self.current_overall)
        )
        self._save_best()

        steps_since_any_improve = 0

        for step in range(1, self.max_steps + 1):
            restart = False

            # --- random restart after patience exhausted ---
            if self.no_improve_count >= self.patience:
                # Jump unconditionally to a large perturbation of best weights.
                # We do NOT apply acceptance testing on a restart — it is a
                # forced escape from the current basin.
                candidate_weights, key = perturb(
                    self.best_weights, self.delta * 5, self.rng
                )
                self.no_improve_count = 0
                restart = True
            else:
                # --- normal perturbation ---
                candidate_weights, key = perturb(
                    self.current_weights, self.delta, self.rng
                )

            _, new_overall, diag = self.curriculum.evaluate_all(candidate_weights)

            old_temp = self.current_overall

            if restart:
                # Always accept restart to advance to the new position.
                accepted = True
                self.current_weights = candidate_weights
                self.current_overall = new_overall
                # (steps_since_any_improve does not reset on a forced restart)
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
                "{}->{}" .format(key[0], key[1]) if key is not None else "—"
            )
            self._print_step(
                step, self.current_overall, self.best_overall,
                key_label, accepted, diag,
            )
            self._log_step(
                step=step,
                accepted=accepted,
                perturbed_key=key_label,
                old_temp=old_temp,
                new_temp=new_overall,
                diagnostic_temps=diag,
                restart=restart,
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

    def _print_step(self, step, current, best, key_label, accepted, diag):
        diag_parts = "  ".join(
            "{}: {:.2f}".format(
                self.curriculum.problem_label(p), t
            )
            for p, t in diag.items()
        )
        status = "✓ accepted" if accepted else "✗ rejected"
        print(
            "step {:4d} | cur {:.4f} | best {:.4f} | {} | {}  [diag: {}]".format(
                step, current, best, status, key_label, diag_parts
            )
        )

    def _log_step(self, step, accepted, perturbed_key, old_temp, new_temp,
                  diagnostic_temps, restart):
        entry = {
            "step": step,
            "accepted": accepted,
            "restart": restart,
            "perturbed_key": perturbed_key,
            "old_temp": old_temp,
            "new_temp": new_temp,
            "best_temp": self.best_overall,
            "diagnostic": {
                "{},{},{}".format(*p): t
                for p, t in diagnostic_temps.items()
            },
        }
        self.step_log.append(entry)

    def _flush_log(self):
        """Append the last log entry to the JSONL file."""
        log_path = os.path.join(self.output_dir, "log.jsonl")
        with open(log_path, "a") as fh:
            fh.write(json.dumps(self.step_log[-1]) + "\n")
