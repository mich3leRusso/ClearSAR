import pandas as pd

df = pd.read_parquet("path/to/file.parquet")
print(df.columns.tolist())
print(df.head())
print(df.shape)
