import pandas as pd

df = pd.read_parquet("CLEARSAR/ClearSAR/catalog.parquet")
print(df.columns.tolist())
print(df.head())
print(df.shape)
