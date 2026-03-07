import pandas as pd

df = pd.read_parquet("CLEA")
print(df.columns.tolist())
print(df.head())
print(df.shape)
