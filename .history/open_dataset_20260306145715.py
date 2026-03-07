import pandas as pd

df = pd.read_parquet("your_file.parquet")
print(df.head())
print(df.info())
print(df.columns.tolist())
