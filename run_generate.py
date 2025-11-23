# run_generate.py
from src.simulate import generate_fleet
out = "data/sample_dataset_generated.csv"
# small but useful sample: 3 devices, ~2 minutes total per device (days is fractional)
df = generate_fleet(num_devices=3, days=0.0015, freq_hz=1, seed=2)
df.to_csv(out, index=False)
print("Wrote", out, "rows=", len(df))
