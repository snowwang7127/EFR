"""Draw Figure 5 directly from the fixed-budget paired contrasts."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def draw(source, output):
    with source.open(newline="", encoding="utf-8-sig") as stream:
        rows = [r for r in csv.DictReader(stream) if (r["A"], r["B"]) == ("entry", "midnight")]
    lags = (0, 4, 8, 10, 12)
    groups = {(r["target"], r["scope"], int(r["lag_h"])): r for r in rows}
    assert len(groups) == len(rows) == 20
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.8, "svg.hashsalt": "ijcis-screening-figure5",
                         "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#667985"})
    fig, axes = plt.subplots(1, 2, figsize=(6.48, 3.48), sharey=True)
    styles = [("all", "All selections", "#295A7D", "o", -0.17),
              ("pending", "Before recorded dispatch", "#AA581D", "s", 0.17)]
    for ax, target, title in zip(axes, ("exact", "recorded"), ("(a)  Elapsed time", "(b)  Recorded flag")):
        ax.set_title(title, loc="left", fontsize=10, weight="bold", color="#263E48")
        ax.axhline(0, color="#475563", ls="--", lw=0.9)
        ax.grid(axis="y", color="#e0e6e7", lw=0.6)
        ax.set_axisbelow(True)
        for scope, _, color, marker, offset in styles:
            for lag in lags:
                r = groups[target, scope, lag]
                n, lo, hi = (int(r[k]) for k in ("delta", "id_lower", "id_upper"))
                assert n == int(r["hit_A"]) - int(r["hit_B"])
                assert lo <= n <= hi
                ax.errorbar(lag + offset, n, yerr=[[n-lo], [hi-n]], fmt=marker, color=color,
                            ms=4.3, mfc="white" if scope=="all" else color, mew=1.1, capsize=2.4, lw=1)
        ax.set(xticks=lags, xlim=(-0.85, 12.85), ylim=(-313, 229), yticks=[-300, -200, -100, 0, 100, 200])
    axes[0].set_ylabel("Confirmed events\n(immediate minus midnight)")
    fig.supxlabel("Assumed dispatch-reporting lag (hours)", fontsize=9, y=0.14)
    handles = [Line2D([], [], marker=m, linestyle="none", color=c, markerfacecolor="white" if k=="all" else c,
                      label=name, markersize=5) for k, name, c, m, _ in styles]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.54, -0.005))
    fig.subplots_adjust(left=.145, right=.983, top=.89, bottom=.275, wspace=.14)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output/"Figure5.svg", metadata={"Date": None})
    fig.savefig(output/"Figure5.png", dpi=330)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paired_csv", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    draw(args.paired_csv, args.output_directory)
