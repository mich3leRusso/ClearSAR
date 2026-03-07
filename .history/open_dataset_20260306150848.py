import pandas as pd

df = pd.read_parquet("CLEARSAR/Clear")
print(df.columns.tolist())
print(df.head())
print(df.shape)
