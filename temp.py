import geopandas as gpd
from shapely.geometry import box

# Create 3 demo fields as rectangles
fields = [
    box(77.5, 28.4, 77.5007, 28.4008),   # Delhi
    box(77.53, 28.41, 77.531, 28.411),   # Delhi2
    box(77.48, 28.39, 77.481, 28.391),   # Delhi3
]
gdf = gpd.GeoDataFrame({'farmer_id': [1, 2, 3]}, geometry=fields, crs="EPSG:4326")
gdf.to_file("demo_fields.geojson", driver="GeoJSON")


import numpy as np
import matplotlib.pyplot as plt

# Simulate NDVI raster: healthy = green, stress = brown
ndvi = np.random.uniform(0.3, 0.9, (100, 100))
ndvi[30:60, 30:70] -= 0.3  # Add a stressed patch

# Colormap: brown-to-green
from matplotlib.colors import LinearSegmentedColormap
brown2green = LinearSegmentedColormap.from_list('b2g', ['#964B00', '#f7e7be', '#93c572', '#208c37'])
plt.imsave("sample_ndvi.png", ndvi, cmap=brown2green)
