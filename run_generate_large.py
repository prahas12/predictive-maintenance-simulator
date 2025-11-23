# run_generate_large.py
from src.simulate import generate_fleet

out = "data/dataset_with_failures.csv"

# More time per device → higher chance of failure occurring
df = generate_fleet(
    num_devices=5,
    days=0.02,          # ~30 minutes total worth of data
    freq_hz=1,
    seed=10
)

df.to_csv(out, index=False)
print("Wrote", out, "rows=", len(df))
print("Failure rows:", df['operational_state'].value_counts().to_dict())
print("Failure types:", df['failure_type'].dropna().unique())
