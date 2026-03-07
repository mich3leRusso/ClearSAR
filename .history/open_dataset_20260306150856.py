import pandas as pd

df = pd.read_parquet("CLEARSAR/ClearSAR/catalog")
print(df.columns.tolist())
print(df.head())
print(df.shape)
