import geopandas as gpd
from shapely.geometry import shape


def tile_intersects_eurocrops(
    product,
    eurocrops_gdf: gpd.GeoDataFrame,
) -> bool:
    """
    Returns True if the Sentinel tile footprint intersects
    at least one EuroCrops polygon.
    """

    # Sentinel tile geometry (WGS84)
    tile_geom_wgs84 = shape(product.footprint)

    tile_gdf = gpd.GeoDataFrame(
        geometry=[tile_geom_wgs84],
        crs="EPSG:4326",
    )

    # Reproject tile footprint to EuroCrops CRS
    tile_gdf = tile_gdf.to_crs(eurocrops_gdf.crs)

    tile_geom = tile_gdf.geometry.iloc[0]

    # Fast spatial index prefilter
    sindex = eurocrops_gdf.sindex
    candidate_idx = list(sindex.intersection(tile_geom.bounds))

    if not candidate_idx:
        return False

    # Exact intersection check
    return bool(eurocrops_gdf.iloc[candidate_idx].intersects(tile_geom).any())
