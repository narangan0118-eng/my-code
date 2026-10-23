
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. Файлын зам
# ==========================================

BASE = Path(__file__).resolve().parent

RESULTS = BASE / "GMVP_All_Portfolios_Results"

PORTFOLIOS = ["High", "Medium", "Low"]
METHODS = ["Var", "Diag", "Corr", "SF", "NLS"]

LATEX_DIR = BASE / "latex"
LATEX_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PDF = LATEX_DIR / "RVR_NT_Three_Portfolios.pdf"
OUTPUT_PNG = LATEX_DIR / "RVR_NT_Three_Portfolios.png"

# ==========================================
# 2. Excel файлуудаас үр дүн унших
# ==========================================

all_results = []

for portfolio in PORTFOLIOS:

    file = (
        RESULTS
        / portfolio
        / f"monthly_rolling_window_{portfolio}.xlsx"
    )

    if not file.exists():
        raise FileNotFoundError(
            f"Excel файл олдсонгүй: {file}"
        )

    df = pd.read_excel(
        file,
        sheet_name="NT_Sensitivity"
    )

    required = {
        "Window",
        "N_T",
        "Method",
        "RVR_Percent"
    }

    if not required.issubset(df.columns):
        raise ValueError(
            f"{portfolio}: шаардлагатай багана дутуу."
        )

    df = df[df["Method"].isin(METHODS)].copy()

    if df.duplicated(["Window", "Method"]).any():
        raise ValueError(
            f"{portfolio}: давхардсан үр дүн байна."
        )

    if set(df["Window"]) != {60, 90, 120}:
        raise ValueError(
            f"{portfolio}: 60, 90, 120 сарын "
            "үр дүн бүрэн биш."
        )

    if len(df) != 15:
        raise ValueError(
            f"{portfolio}: 15 үр дүн шаардлагатай."
        )

    df["Portfolio"] = portfolio

    all_results.append(df)

results = pd.concat(
    all_results,
    ignore_index=True
)

# ==========================================
# 3. График байгуулах
# ==========================================

fig, axes = plt.subplots(
    1, 3,
    figsize=(15, 4.8),
    sharex=True,
    sharey=True
)

markers = {
    "Var": "o",
    "Diag": "s",
    "Corr": "^",
    "SF": "D",
    "NLS": "v"
}

for ax, portfolio in zip(axes, PORTFOLIOS):

    subset = results[
        results["Portfolio"] == portfolio
    ]

    for method in METHODS:

        data = subset[
            subset["Method"] == method
        ].sort_values("N_T")

        ax.plot(
            data["N_T"],
            data["RVR_Percent"],
            marker=markers[method],
            linewidth=1.7,
            markersize=5,
            label=method
        )

    ax.axhline(
        y=0,
        color="black",
        linestyle="--",
        linewidth=0.9
    )

    ax.set_title(
        f"{portfolio} Correlation",
        fontsize=12
    )

    ax.set_xlabel(
        "N/T Ratio",
        fontsize=11
    )

    ax.set_xticks([
        10 / 120,
        10 / 90,
        10 / 60
    ])

    ax.set_xticklabels([
        "0.0833",
        "0.1111",
        "0.1667"
    ])

    ax.grid(
        True,
        alpha=0.25
    )

axes[0].set_ylabel(
    "Relative Variance Reduction (%)",
    fontsize=11
)

# Нэг ерөнхий тайлбар
handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=5,
    frameon=False,
    bbox_to_anchor=(0.5, 0.01)
)

fig.tight_layout(
    rect=[0, 0.10, 1, 1]
)

# ==========================================
# 4. PDF болон PNG хадгалах
# ==========================================

fig.savefig(
    OUTPUT_PDF,
    bbox_inches="tight"
)

fig.savefig(
    OUTPUT_PNG,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

print("График амжилттай байгуулагдлаа.")
print("PDF:", OUTPUT_PDF)
print("PNG:", OUTPUT_PNG)
