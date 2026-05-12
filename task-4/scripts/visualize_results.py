import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_PATH = "../benchmarks/benchmark_results.csv"

OUTPUT_DIR = Path("../results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------
# Load Results
# ---------------------------------------

df = pd.read_csv(RESULTS_PATH)

# ---------------------------------------
# Latency Table
# ---------------------------------------

latency_columns = [
    "average_latency_ms",
    "p50_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms"
]

latency_df = df[latency_columns]

latency_table_path = OUTPUT_DIR / "latency_table.csv"
latency_df.to_csv(latency_table_path, index=False)

print(f"Latency table saved to: {latency_table_path}")

# ---------------------------------------
# FPS Chart
# ---------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    df["runtime"],
    df["fps"]
)

plt.xlabel("Runtime")
plt.ylabel("FPS")
plt.title("FPS Comparison")

fps_chart_path = OUTPUT_DIR / "fps_comparison.png"

plt.savefig(fps_chart_path)

print(f"FPS chart saved to: {fps_chart_path}")

# ---------------------------------------
# Percentile Latency Chart
# ---------------------------------------

percentiles = [
    "p50_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms"
]

values = [df[p].iloc[0] for p in percentiles]

plt.figure(figsize=(8, 5))

plt.bar(percentiles, values)

plt.xlabel("Percentile")
plt.ylabel("Latency (ms)")
plt.title("Latency Percentiles")

latency_chart_path = OUTPUT_DIR / "latency_percentiles.png"

plt.savefig(latency_chart_path)

print(f"Latency percentile chart saved to: {latency_chart_path}")