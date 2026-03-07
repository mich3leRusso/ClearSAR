import pandas as pd

df = pd.read_parquet("CLEARSAR/ClearSAR/catalog.parquet")
print(df.head())
print(df.info())
print(df.columns.tolist())
