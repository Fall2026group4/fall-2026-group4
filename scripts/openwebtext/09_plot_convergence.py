
import matplotlib.pyplot as plt
from pathlib import Path

architectures = ["JumpReLU", "TopK", "ReLU"]
mse_1000 = [1.68032, 1.21845, 1.66331]
mse_15000 = [0.62183, 0.66413, 0.78255]

rows = []
for name, before, after in zip(architectures, mse_1000, mse_15000):
    improvement = (before - after) / before * 100
    rows.append([
        name,
        f"{before:.5f}",
        f"{after:.5f}",
        f"{improvement:.1f}%"
    ])

fig, ax = plt.subplots(figsize=(11, 3.5))
ax.axis("off")

table = ax.table(
    cellText=rows,
    colLabels=[
        "Architecture",
        "1,000-step MSE",
        "15,000-step MSE",
        "Total Improvement"
    ],
    cellLoc="center",
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2.2)

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("#D1D5DB")
    if row == 0:
        cell.set_facecolor("#1F2937")
        cell.set_text_props(color="white", weight="bold")
    else:
        cell.set_facecolor("#F9FAFB" if row % 2 else "white")

ax.set_title(
    "SAE Reconstruction Improvement",
    fontsize=16,
    fontweight="bold",
    pad=20
)

output = Path("results/openwebtext/figures")
output.mkdir(parents=True, exist_ok=True)

plt.savefig(
    output / "sae_convergence_table.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()
print("Table saved successfully!")
