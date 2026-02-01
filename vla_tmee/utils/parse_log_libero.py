import re
import os
import pandas as pd

file_path = "your_path"

# Read the log file
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# Extract all per-task success rates
task_rates = re.findall(r"Current task success rate:\s*([\d.]+)", text)
task_rates = [float(x) for x in task_rates]

# Extract the final overall success rate
total_rates = re.findall(r"Current total success rate:\s*([\d.]+)", text)
final_total_rate = float(total_rates[-1]) if total_rates else None

# Organize results into a single-row dictionary
data = {}
for i, rate in enumerate(task_rates, start=1):
    data[f"task_{i}"] = [rate]

data["final_total_success_rate"] = [final_total_rate]

# Convert to DataFrame
df = pd.DataFrame(data)

# Save CSV to the same directory as the log file
output_dir = os.path.dirname(file_path)
output_path = os.path.join(output_dir, "success_rates_horizontal.csv")
df.to_csv(output_path, index=False)

print(f"✅ Saved success rates to {output_path}")
