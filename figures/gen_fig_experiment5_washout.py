"""Reproducible, measured fixed-validation curves (no smoothed or invented points)."""

def plot(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    with plt.rc_context({"font.family": "serif", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42}):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for ax, split in zip(axes, ("confirmation", "discovery")):
            data = [r for r in rows if r["split"] == split]
            x = [r["offset"] for r in data]
            for name, key, color, marker in (("A−B", "AB", "#0072B2", "o"), ("A−C", "AC", "#D55E00", "s")):
                y = np.array([r["A_minus_B" if key == "AB" else "A_minus_C"] for r in data])
                low = np.array([r[f"{key}_ci95_low"] for r in data])
                high = np.array([r[f"{key}_ci95_high"] for r in data])
                ax.plot(x, y, marker=marker, color=color, label=name)
                ax.vlines(x, low, high, color=color, alpha=0.7)
            ax.axhline(0, color="black", lw=0.8)
            ax.axhspan(-0.001, 0.001, color="gray", alpha=0.12, label="±0.001 diagnostic margin")
            ax.set_xscale("symlog", linthresh=2)
            ax.set_xlim(-0.08, 110)
            ax.set_xticks(x, labels=[str(v) for v in x])
            ax.set(xlabel="Optimizer updates after intervention (symlog)", ylabel="Fixed-validation ΔNLL (nats/token)", title=split.capitalize())
            ax.grid(alpha=0.15); ax.legend(fontsize=8)
        fig.suptitle("Experiment 5: negative favors A; pointwise paired 95% intervals")
        fig.savefig(output / "fig_washout.pdf")
        fig.savefig(output / "fig_washout.png", dpi=300)
        plt.close(fig)
