import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
json_path = RESULTS_DIR / "hybrid_driver_drowsiness_results.json"

with open(json_path) as f:
    data = json.load(f)

summary = data.get("session_summary", {})


plt.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "default"
)
plt.rcParams["font.size"] = 11


categories = [
    "CNN Classifier\n(Spatial)",
    "EAR Threshold\n(Eye Closure)",
    "MAR Threshold\n(Yawning)",
]
counts = [
    summary.get("cnn_trigger_count", 0),
    summary.get("ear_trigger_count", 0),
    summary.get("mar_trigger_count", 0),
]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(categories, counts, color=["#1f77b4", "#ff7f0e", "#2ca02c"], width=0.5)
ax.set_ylabel("Trigger Count (Frames)")
ax.set_title("Live Session: Hybrid Component Trigger Distribution")
ax.set_ylim(0, max(counts) * 1.2 if max(counts) > 0 else 10)

for bar in bars:
    height = bar.get_height()
    ax.annotate(
        f"{height:,}",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontweight="bold",
    )

plt.tight_layout()
plt.savefig(RESULTS_DIR / "live_hybrid_component_triggers.png", dpi=300)
plt.close()

# Geometric Means vs Thresholds
metrics = ["EAR (Eye Aspect Ratio)", "MAR (Mouth Aspect Ratio)"]
means = [summary.get("mean_ear", 0.0), summary.get("mean_mar", 0.0)]
thresholds = [0.21, 0.50]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width / 2, means, width, label="Session Mean Value", color="#aec7e8")
ax.bar(x + width / 2, thresholds, width, label="Decision Threshold", color="#ff9896")

ax.set_ylabel("Ratio Value")
ax.set_title("Live Session: Geometric Feature Means vs. Decision Thresholds")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend(loc="upper right")
ax.set_ylim(0, 1.0)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "live_geometric_thresholds.png", dpi=300)
plt.close()

print("Report-ready live charts successfully generated in evaluation/results/")
