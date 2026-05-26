import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

plt.rcParams["font.family"] = "Asap"

data = {
    2: {
        "bp": {"score": (24.4543, 0.0079), "perfect": (69.58, 0.33)},
        "ues": [
            {"score": (24.6142, 0.0080), "perfect": (75.72, 0.43)},
            {"score": (24.6537, 0.0074), "perfect": (77.53, 0.42)},
        ],
        "sota_score": 24.62,
    },
    3: {
        "bp": {"score": (24.6118, 0.0080), "perfect": (77.61, 0.29)},
        "ues": [
            {"score": (24.7804, 0.0063), "perfect": (84.77, 0.36)},
            {"score": (24.8366, 0.0049), "perfect": (87.81, 0.33)},
            {"score": (24.8634, 0.0046), "perfect": (89.90, 0.30)},
        ],
        "sota_score": 24.76,
    },
    4: {
        "bp": {"score": (24.4658, 0.0084), "perfect": (68.52, 0.33)},
        "ues": [
            {"score": (24.6832, 0.0068), "perfect": (77.65, 0.42)},
            {"score": (24.7755, 0.0062), "perfect": (83.50, 0.37)},
            {"score": (24.8212, 0.0050), "perfect": (86.13, 0.35)},
            {"score": (24.8520, 0.0046), "perfect": (88.40, 0.32)},
        ],
        "sota_score": 24.59,
    },
    5: {
        "bp": {"score": (23.4854, 0.0119), "perfect": (26.81, 0.31)},
        "ues": [
            {"score": (24.0076, 0.0106), "perfect": (39.19, 0.49)},
            {"score": (24.2125, 0.0097), "perfect": (47.43, 0.50)},
            {"score": (24.3400, 0.0087), "perfect": (53.94, 0.50)},
            {"score": (24.3932, 0.0085), "perfect": (56.84, 0.50)},
            {"score": (24.4097, 0.0091), "perfect": (58.05, 0.49)},
        ],
        "sota_score": 24.09,
    },
}

TITLE_FS = 16
LABEL_FS = 14
TICK_FS = 14
LEGEND_FS = 14
ANNOT_FS = 14


def plot_metric(ax, n_players, d, metric, ylabel, annot_fmt, sota, show_delta=False):
    bp_y, bp_e = d["bp"][metric]
    ues_x = [0] + list(range(1, len(d["ues"]) + 1))
    ues_y = [bp_y] + [e[metric][0] for e in d["ues"]]
    ues_e = [bp_e] + [e[metric][1] for e in d["ues"]]

    ues_line = ax.errorbar(
        ues_x,
        ues_y,
        yerr=ues_e,
        marker="o",
        capsize=3,
        markersize=5,
        label="Ataraxos",
        color="tab:blue",
        linewidth=1.8,
    )
    for x, y in zip(ues_x, ues_y):
        if x == 0:
            xytext, ha = (8, -4), "left"
        elif x == 1:
            xytext, ha = (4, -12), "left"
        elif x == 2:
            xytext, ha = (4, -16), "left"
        elif x == 3:
            xytext, ha = (2, -16), "left"
        else:
            xytext, ha = (0, -16), "center"

        if x == n_players:
            xytext, ha = (-4, -20), "center"
        ax.annotate(
            annot_fmt.format(y),
            (x, y),
            textcoords="offset points",
            xytext=xytext,
            ha=ha,
            fontsize=ANNOT_FS,
            color="tab:blue",
        )

    handles = [ues_line]
    if sota is not None:
        sota_line = ax.axhline(
            sota, linestyle="--", color="gray", label=f"Previous SOTA ({sota})"
        )
        handles.append(sota_line)

    ax.set_xlabel("Number of Search Players", fontsize=LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    ax.set_title(f"{n_players}-Player Hanabi", fontsize=TITLE_FS)
    ax.set_xticks(list(range(0, n_players + 1)))
    ax.set_xlim(-0.25, n_players + 0.25)
    y_range = max(ues_y) - bp_y
    ax.set_ylim(bp_y - 0.05 * y_range - 0.01, max(ues_y) + 0.08 * y_range + 0.01)
    if metric == "perfect" and n_players == 4:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if show_delta:
        baseline = ues_y[0]
        ceiling = ues_y[-1]
        delta = ceiling - baseline
        ax.hlines(
            ceiling,
            0,
            n_players,
            linestyle="--",
            color="tab:orange",
            alpha=0.6,
            linewidth=1.2,
        )
        arrow_x = -0.05
        ax.annotate(
            "",
            xy=(arrow_x, ceiling),
            xytext=(arrow_x, baseline),
            arrowprops=dict(arrowstyle="<->", color="tab:orange", lw=1.8),
        )
        ax.text(
            arrow_x + 0.1,
            (baseline + ceiling) / 2,
            f"+{delta:.1f}",
            color="tab:orange",
            fontsize=ANNOT_FS,
            ha="left",
            va="center",
            fontweight="bold",
        )

    leg = ax.legend(
        handles=handles,
        loc="lower right",
        fontsize=LEGEND_FS,
        frameon=False,
        markerfirst=False,
    )
    leg._legend_box.align = "right"
    for t in leg.get_texts():
        t.set_ha("right")
    ax.grid(alpha=0.3)


def make_figure(metric, ylabel, annot_fmt, use_sota, show_delta, out_path):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8))
    for ax, n_players in zip(axes, [2, 3, 4, 5]):
        d = data[n_players]
        plot_metric(
            ax,
            n_players,
            d,
            metric=metric,
            ylabel=ylabel,
            annot_fmt=annot_fmt,
            sota=d["sota_score"] if use_sota else None,
            show_delta=show_delta,
        )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    pdf_path = out_path.rsplit(".", 1)[0] + ".pdf"
    plt.savefig(pdf_path)
    plt.close(fig)
    print(f"saved {out_path} and {pdf_path}")


make_figure("score", "Score", "{:.2f}", True, False, "figs/plot_search_score.png")
make_figure(
    "perfect",
    "Perfect Games (%)",
    "{:.1f}",
    False,
    True,
    "figs/plot_search_perfect.png",
)
