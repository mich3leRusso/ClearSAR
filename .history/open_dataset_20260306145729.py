import pandas as pd

df = pd.read_parquet("C")
print(df.head())
print(df.info())
print(df.columns.tolist())
