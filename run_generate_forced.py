# run_generate_forced.py
from src.simulate import generate_fleet
import pandas as pd

out = "data/dataset_forced_failures.csv"

# Override defaults by making more time and higher failure probability
df = generate_fleet(
    num_devices=5,
    days=0.05,         # around ~1.2 hours worth of data per device (good for failure)
    freq_hz=1,
    seed=42            # different seed
)

# FORCE failures if none were generated (safety)
if df['operational_state'].max() == 0:
    print("⚠️ No natural failures found. Forcing synthetic failures.")

    # Force last 200 rows of each device to degraded (1) then failed (2)
    df_forced = []

    for device_id in df['device_id'].unique():
        part = df[df['device_id'] == device_id].copy()
        if len(part) > 200:
            part.iloc[-200:-100, part.columns.get_loc('operational_state')] = 1  # degraded
            part.iloc[-100:, part.columns.get_loc('operational_state')] = 2      # failed
            part.iloc[-100:, part.columns.get_loc('failure_type')] = 'forced_failure'
        df_forced.append(part)

    df = pd.concat(df_forced)

df.to_csv(out, index=False)

print("Wrote", out, "rows=", len(df))
print("State counts:", df['operational_state'].value_counts().to_dict())
print("Failure types:", df['failure_type'].dropna().unique())
