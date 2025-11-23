# check_data.py
import pandas as pd

path = "data/sample_dataset.csv"
df = pd.read_csv(path)

print("Loaded file:", path)
print("Rows:", len(df))
print("Columns:", list(df.columns))
print("\nFirst 5 rows:")
print(df.head(5).to_string(index=False))
