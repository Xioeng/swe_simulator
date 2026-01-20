import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

from utils import interpolate_gebco_on_grid


def main():
    nc_path = "../data/gebco_2025_n25.9288_s25.6527_w-80.2016_e-80.0642.nc"

    # Define a simple lon/lat grid covering the GEBCO tile
    lon_min, lon_max = -80.2016, -80.0642
    lat_min, lat_max = 25.6527, 25.9288

    nx, ny = 200, 200
    lon = np.linspace(lon_min, lon_max, nx)
    lat = np.linspace(lat_min, lat_max, ny)
    X, Y = np.meshgrid(lon, lat, indexing="ij")  # X=lon, Y=lat

    Z = interpolate_gebco_on_grid(nc_path, X, Y)
    Z[Z > 0.2] = np.nan

    fig, ax = plt.subplots(
        figsize=(6, 5), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    im = ax.pcolormesh(X, Y, Z, shading="auto", cmap="viridis", alpha=0.4)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    # Add Cartopy background image
    import cartopy.feature as cfeature

    # Add satellite imagery from Stamen (terrain-background)
    import cartopy.io.img_tiles as cimgt

    google_tiles = cimgt.GoogleTiles(style="only_streets")
    ax.add_image(
        google_tiles,
        14,  # zoom level, adjust as needed
        interpolation="bilinear",
    )
    # ax.add_feature(cfeature.OCEAN, zorder=0)
    # ax.add_feature(cfeature.COASTLINE, zorder=1)
    ax.set_title("GEBCO bathymetry (elevation)")
    # fig.colorbar(im, ax=ax, label="Elevation (m)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
