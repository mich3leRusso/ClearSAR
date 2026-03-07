import pandas as pd
import geopandas as gpd

gdf = gpd.read_parquet("your_file.parquet")
print(gdf.crs)         # coordinate reference system
print(gdf.geometry)    # spatial geometry column
gdf.plot()             # quick visual of the coverage area


df = pd.read_parquet("CLEARSAR/ClearSAR/catalog.parquet")
print(df.head())
print(df.info())
print(df.columns.tolist())
