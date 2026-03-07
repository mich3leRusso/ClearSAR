import pandas as pd

df = pd.read_parquet("C")
print(df.columns.tolist())
print(df.head())
print(df.shape)
