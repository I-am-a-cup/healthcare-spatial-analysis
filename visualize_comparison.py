"""Render the model comparison from the shared saved metrics."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent
with (ROOT / "results/model_comparison.csv").open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f, delimiter=";"))
names = [r["model"] for r in rows]
fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), layout="constrained")
for ax, field, title, fmt in zip(axes, ("AIC","RMSE_in_sample","Moran_I_innovation"),
    ("AIC (меньше — лучше)", "Внутривыборочная RMSE, лет", "Moran I инноваций"), (".2f",".3f",".3f")):
    values = [float(r[field]) for r in rows]
    bars = ax.bar(names, values, color=["#8793a2","#527e9c","#38786c"])
    ax.bar_label(bars, labels=[format(x,fmt) for x in values], padding=4)
    ax.set_title(title, fontsize=11)
    ax.margins(y=.22)
    ax.axhline(0, color="#666666",linewidth=.6)
    ax.spines[["right","top"]].set_visible(False)
    ax.grid(axis="y", alpha=.18)
fig.suptitle("Сопоставление OLS, SAR и SEM · 85 регионов, 2021",fontsize=14)
fig.savefig(ROOT / "maps/06_model_comparison.png", dpi=200)
plt.close(fig)
