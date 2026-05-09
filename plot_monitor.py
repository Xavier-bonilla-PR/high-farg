#!/usr/bin/env python3
"""
Visualise weights/monitor.jsonl produced by the Slipnet weight learner.

Panels
------
1. Mean temperature & best  — the optimisation target over time
2. Std (σ)                  — overall σ + per-diagnostic σ
3. Failures                 — % of trials finishing with temp > 60
4. Preferred-answer rate    — % of trials producing a celebrated answer
5. Diagnostic std detail    — σ for each of the two diagnostic problems
6. Answer distribution      — heatmap: top-N answers × step (colour = % of trials)

Usage
-----
    python plot_monitor.py                          # reads weights/monitor.jsonl
    python plot_monitor.py path/to/monitor.jsonl
    python plot_monitor.py --output report.png      # save instead of showing
    python plot_monitor.py --top-answers 12         # widen the answer heatmap
"""

import argparse
import json
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import numpy as np


# ── colour palette ─────────────────────────────────────────────────────────
C_MEAN    = "#2196F3"   # blue
C_BEST    = "#0D47A1"   # dark blue
C_STD     = "#9C27B0"   # purple
C_FAIL    = "#F44336"   # red
C_PREF    = "#4CAF50"   # green
C_DIAG    = ["#FF5722", "#FF9800"]   # orange shades for the 2 diagnostic problems
C_ACCEPT  = "#00C853"   # bright green for accepted-step markers


# ── data loading ───────────────────────────────────────────────────────────

def load_monitor(path):
    entries = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    if not entries:
        sys.exit("monitor.jsonl is empty — run the learner first.")
    return entries


def extract(entries):
    """Pull every time-series we need out of the entry list."""
    steps    = [e["step"] for e in entries]
    accepted = [e["accepted"] for e in entries]

    overall  = [e["overall"] for e in entries]
    mean     = [o["mean"]         for o in overall]
    std      = [o["std"]          for o in overall]
    failures = [o["failure_rate"] * 100 for o in overall]
    pref     = [o.get("preferred_rate", None) for o in overall]
    best     = []                   # running best mean
    cur_best = float("inf")
    for m, a in zip(mean, accepted):
        if a and m < cur_best:
            cur_best = m
        best.append(cur_best if cur_best < float("inf") else m)

    diag_keys = list(entries[0]["diagnostic"].keys())
    diag_std  = {k: [e["diagnostic"][k]["std"]  for e in entries] for k in diag_keys}
    diag_mean = {k: [e["diagnostic"][k]["mean"] for e in entries] for k in diag_keys}
    diag_pref = {
        k: [e["diagnostic"][k].get("preferred_rate", None) for e in entries]
        for k in diag_keys
    }

    # answer distribution: collect top-N answers over all steps
    n_trials   = overall[0].get("n_trials", 45)
    step_dists = [o["answer_distribution"] for o in overall]
    total_cnt  = defaultdict(int)
    for dist in step_dists:
        for ans, cnt in dist.items():
            total_cnt[ans] += cnt

    return dict(
        steps=steps, accepted=accepted,
        mean=mean, std=std, failures=failures, pref=pref, best=best,
        diag_keys=diag_keys, diag_std=diag_std, diag_mean=diag_mean,
        diag_pref=diag_pref, step_dists=step_dists,
        n_trials=n_trials, total_cnt=total_cnt,
    )


def build_answer_matrix(step_dists, top_answers, n_trials):
    """Return (n_steps × n_answers) matrix of percentages."""
    mat = np.zeros((len(step_dists), len(top_answers)))
    for i, dist in enumerate(step_dists):
        for j, ans in enumerate(top_answers):
            mat[i, j] = dist.get(ans, 0) / n_trials * 100
    return mat


# ── plotting helpers ────────────────────────────────────────────────────────

def mark_accepted(ax, steps, accepted, values, color=C_ACCEPT, size=18):
    """Scatter-plot dots only on accepted steps."""
    xs = [s for s, a in zip(steps, accepted) if a]
    ys = [v for v, a in zip(values, accepted) if a]
    ax.scatter(xs, ys, color=color, s=size, zorder=5)


def short_diag_label(key):
    """'abc,abd,kji' → 'kji'  (just the target string)."""
    return key.split(",")[2]


# ── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("monitor", nargs="?", default="weights/monitor.jsonl",
                        help="Path to monitor.jsonl (default: weights/monitor.jsonl)")
    parser.add_argument("--output", default=None,
                        help="Save figure to this file instead of displaying it")
    parser.add_argument("--top-answers", type=int, default=10,
                        help="How many distinct answers to show in the heatmap (default: 10)")
    args = parser.parse_args()

    entries = load_monitor(args.monitor)
    d = extract(entries)

    top_answers = sorted(d["total_cnt"], key=lambda a: -d["total_cnt"][a])[: args.top_answers]
    ans_matrix  = build_answer_matrix(d["step_dists"], top_answers, d["n_trials"])

    # ── figure layout ───────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 13))
    fig.patch.set_facecolor("#FAFAFA")
    fig.suptitle("Slipnet Learner — Monitor Dashboard", fontsize=15,
                 fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(
        3, 2,
        figure=fig,
        height_ratios=[1, 1, 1.2],
        hspace=0.48,
        wspace=0.32,
    )

    ax_mean = fig.add_subplot(gs[0, 0])
    ax_std  = fig.add_subplot(gs[0, 1])
    ax_fail = fig.add_subplot(gs[1, 0])
    ax_pref = fig.add_subplot(gs[1, 1])
    ax_diag = fig.add_subplot(gs[2, 0])
    ax_ans  = fig.add_subplot(gs[2, 1])

    steps = d["steps"]

    # ── 1. Mean temperature ─────────────────────────────────────────────────
    ax_mean.plot(steps, d["mean"], color=C_MEAN, linewidth=1.4, label="current mean")
    ax_mean.plot(steps, d["best"], color=C_BEST,  linewidth=1.0,
                 linestyle="--", label="running best")
    mark_accepted(ax_mean, steps, d["accepted"], d["mean"])
    ax_mean.set_title("Mean Temperature", fontweight="bold")
    ax_mean.set_xlabel("Step")
    ax_mean.set_ylabel("Temperature")
    ax_mean.legend(fontsize=8)
    ax_mean.grid(True, alpha=0.25)

    # ── 2. Overall std (σ) + per-diagnostic σ ───────────────────────────────
    ax_std.plot(steps, d["std"], color=C_STD, linewidth=1.6, label="overall σ")
    for (k, vals), col in zip(d["diag_std"].items(), C_DIAG):
        ax_std.plot(steps, vals, color=col, linewidth=1.0,
                    linestyle="--", label="{} σ".format(short_diag_label(k)))
    mark_accepted(ax_std, steps, d["accepted"], d["std"], color=C_ACCEPT)
    ax_std.set_title("Standard Deviation (σ)", fontweight="bold")
    ax_std.set_xlabel("Step")
    ax_std.set_ylabel("σ")
    ax_std.legend(fontsize=8)
    ax_std.grid(True, alpha=0.25)

    # ── 3. Failures ─────────────────────────────────────────────────────────
    ax_fail.plot(steps, d["failures"], color=C_FAIL, linewidth=1.4)
    ax_fail.fill_between(steps, 0, d["failures"], color=C_FAIL, alpha=0.12)
    mark_accepted(ax_fail, steps, d["accepted"], d["failures"], color=C_ACCEPT)
    ax_fail.set_title("Failure Rate  (final temp > 60)", fontweight="bold")
    ax_fail.set_xlabel("Step")
    ax_fail.set_ylabel("% of trials")
    ax_fail.set_ylim(bottom=0)
    ax_fail.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax_fail.grid(True, alpha=0.25)

    # ── 4. Preferred-answer rate ─────────────────────────────────────────────
    has_pref = any(p is not None for p in d["pref"])
    if has_pref:
        pref_pct = [p * 100 if p is not None else float("nan") for p in d["pref"]]
        ax_pref.plot(steps, pref_pct, color=C_PREF, linewidth=1.4, label="overall")
        for (k, vals), col in zip(d["diag_pref"].items(), C_DIAG):
            pct = [v * 100 if v is not None else float("nan") for v in vals]
            ax_pref.plot(steps, pct, color=col, linewidth=1.0,
                         linestyle="--", label=short_diag_label(k))
        mark_accepted(ax_pref, steps, d["accepted"], pref_pct, color=C_ACCEPT)
        ax_pref.set_ylim(0, 105)
        ax_pref.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
        ax_pref.set_title("Preferred-Answer Rate", fontweight="bold")
        ax_pref.set_xlabel("Step")
        ax_pref.set_ylabel("% preferred")
        ax_pref.legend(fontsize=8)
        ax_pref.grid(True, alpha=0.25)
    else:
        ax_pref.text(0.5, 0.5, "No preferred-answer data\n(run with --answer-weight > 0)",
                     ha="center", va="center", transform=ax_pref.transAxes,
                     fontsize=10, color="grey")
        ax_pref.set_title("Preferred-Answer Rate", fontweight="bold")

    # ── 5. Diagnostic std comparison ─────────────────────────────────────────
    for (k, vals), col in zip(d["diag_std"].items(), C_DIAG):
        ax_diag.plot(steps, vals, color=col, linewidth=1.4,
                     label="{} σ".format(short_diag_label(k)))
        ax_diag.fill_between(steps, 0, vals, color=col, alpha=0.08)
    ax_diag.set_title("Diagnostic Problem σ", fontweight="bold")
    ax_diag.set_xlabel("Step")
    ax_diag.set_ylabel("σ")
    ax_diag.legend(fontsize=8)
    ax_diag.grid(True, alpha=0.25)

    # ── 6. Answer distribution heatmap ───────────────────────────────────────
    im = ax_ans.imshow(
        ans_matrix.T,
        aspect="auto",
        cmap="YlOrRd",
        interpolation="nearest",
        vmin=0,
        vmax=min(100, ans_matrix.max() * 1.1 + 1),
    )
    ax_ans.set_yticks(range(len(top_answers)))
    ax_ans.set_yticklabels(top_answers, fontsize=8, fontfamily="monospace")
    ax_ans.set_xlabel("Step")
    ax_ans.set_title("Answer Distribution — top {}".format(args.top_answers),
                     fontweight="bold")

    # x-axis: show a reasonable number of step labels
    n_steps = len(steps)
    tick_every = max(1, n_steps // 10)
    tick_positions = list(range(0, n_steps, tick_every))
    ax_ans.set_xticks(tick_positions)
    ax_ans.set_xticklabels([steps[i] for i in tick_positions], rotation=45, ha="right")

    cb = fig.colorbar(im, ax=ax_ans, shrink=0.85, pad=0.02)
    cb.set_label("% of trials", fontsize=8)

    # Mark restart steps with a vertical line across all axes
    restart_steps = [e["step"] for e in entries if e.get("restart", False)]
    for ax in [ax_mean, ax_std, ax_fail, ax_pref, ax_diag]:
        for rs in restart_steps:
            ax.axvline(rs, color="#795548", alpha=0.35, linewidth=0.8, linestyle=":")

    # ── summary stats in figure footer ──────────────────────────────────────
    n_accepted = sum(d["accepted"])
    accept_rate = n_accepted / len(steps) * 100
    summary = (
        "  {} steps  |  {} accepted ({:.0f}%)  |  {} restarts  |"
        "  best mean {:.2f}  |  final pref {:.0f}%"
    ).format(
        len(steps),
        n_accepted,
        accept_rate,
        len(restart_steps),
        min(d["mean"]),
        (d["pref"][-1] or 0) * 100 if has_pref else float("nan"),
    )
    fig.text(0.5, 0.005, summary, ha="center", fontsize=8, color="#555555")

    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print("Saved to {}".format(args.output))
    else:
        plt.show()


if __name__ == "__main__":
    main()
