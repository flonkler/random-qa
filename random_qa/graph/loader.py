from geopandas import GeoDataFrame, GeoSeries
from pandas import DataFrame, Series, read_csv, isna
import pyproj
import geopy.distance
from geographiclib.geodesic import Geodesic
import numpy as np
import json
import os
import itertools

from shapely.geometry import Point, Polygon, box

from random_qa.utils import CRS_WGS84_GPS, CRS_ETRS89_UTM33N, transform_bbox, find_largest_intersection
from random_qa.graph.node import (
    RadioAccessTechnologyEnum, OperatorEnum, BandEnum, ConstructionEnum, ManufacturerEnum, ConnectionEnum
)

def load_regions_dataframe(data_dir: str) -> GeoDataFrame:
    with open(os.path.join(data_dir, "regions.json"), "r") as f:
        geojson = json.load(f)

    gdf = GeoDataFrame.from_features(geojson, crs=CRS_ETRS89_UTM33N)
    gdf = gdf.drop(columns=["FID", "OT"]).rename(columns={"Name": "name"})
    gdf = gdf.assign(
        id=gdf.name.apply(lambda n: f"Region:{n}"),
        area=gdf.geometry.area / 1e6
    )
    inhabitants = read_csv(os.path.join(data_dir, "population.csv"))
    inhabitants = inhabitants.set_index("Gebiet")
    inhabitants = inhabitants[inhabitants.Sachmerkmal == "Einwohner insgesamt"]
    gdf = gdf.join(inhabitants["2023"].rename("population_count"), on="name", how="left")
    gdf["population_density"] = gdf["population_count"] / gdf["area"]
    return gdf.set_index("id")


def load_sites_dataframe(data_dir: str) -> GeoDataFrame:
    df = read_csv(os.path.join(data_dir, "sites.csv"), sep=";")

    def _parse_operators(operators):
        return list(map(lambda x: f"Operator:{OperatorEnum(x)}", operators.split(",")))

    data = {
        "name": df["site_id"],
        "geometry": df.apply(lambda row: Point(*tuple(map(float, row["coordinates"].split(",")))[::-1]), axis=1),
        "coordinates": df["coordinates"],
        "operated_by": df["operators"].apply(_parse_operators),
        "construction": df["construction"].apply(ConstructionEnum)
    }
    gdf = GeoDataFrame(data, crs=CRS_WGS84_GPS).to_crs(CRS_ETRS89_UTM33N)
    gdf["id"] = "Site:" + gdf["name"].astype(str)
    return gdf.set_index("id")


def load_mobile_antennas_dataframe(data_dir: str, gdf_sites: GeoDataFrame) -> GeoDataFrame:
    df = read_csv(os.path.join(data_dir, "antennas.csv"), sep=";")
    df["installed_at"] = "Site:" + df["site_id"].astype(str)
    df = df.join(gdf_sites, on="installed_at", rsuffix="_s")

    def _offset_antenna_position(point: Point, orientation: float, offset: float = 5):
        if isna(orientation):
            return point
        # Offset position of antenna based on its orientation
        transformer = pyproj.Transformer.from_crs(CRS_ETRS89_UTM33N, CRS_WGS84_GPS)
        coords = geopy.distance.distance(meters=offset).destination(
            point=transformer.transform(point.x, point.y),
            bearing=orientation
        )
        return Point(*transformer.transform(coords.latitude, coords.longitude, direction="INVERSE"))

    gdf = GeoDataFrame({
        "id": "MobileAntenna:" + df["antenna_id"],
        "name": df["antenna_id"],
        "geometry": df.apply(lambda row: _offset_antenna_position(row.geometry, row.orientation), axis=1),
        "orientation": df["orientation"],
        "operated_by": "Operator:" + df["operator"].apply(OperatorEnum),
        "produced_by": "Manufacturer:" + df["manufacturer"].apply(ManufacturerEnum),
        "site_construction": df["construction"],
        "installed_at": df["installed_at"]
    }, crs=CRS_ETRS89_UTM33N)
    return gdf.set_index("id")


def load_microwave_antennas_dataframe(gdf_sites: GeoDataFrame, df_manufacturers: DataFrame) -> GeoDataFrame:
    data = []
    def _compute_orientation(start: Point, end: Point) -> float:
        transformer = pyproj.Transformer.from_crs(CRS_ETRS89_UTM33N, CRS_WGS84_GPS)
        res = Geodesic.WGS84.Inverse(*transformer.transform(start.x, start.y), *transformer.transform(end.x, end.y))
        return np.round((res["azi1"] + 360) % 360, 1)

    for site_id, site in gdf_sites[gdf_sites.connection == ConnectionEnum.MICROWAVE].iterrows():
        manufacturer = df_manufacturers.name.sample(1, weights=df_manufacturers.market_share).iloc[0]
        parent_site_id = site.connection_provided_by
        parent_site = gdf_sites.loc[parent_site_id]        
        name_a = f"RIFU{len(data) + 1:06d}"
        name_b = f"RIFU{len(data) + 2:06d}"
        data += [
            {
                "id": f"MicrowaveAntenna:{name_a}",
                "name": name_a,
                "geometry": site.geometry,
                "orientation": _compute_orientation(site.geometry, parent_site.geometry),
                "installed_at": site_id,
                "produced_by": f"Manufacturer:{manufacturer}",
                "connected_with": f"MicrowaveAntenna:{name_b}"
            },
            {
                "id": f"MicrowaveAntenna:{name_b}",
                "name": name_b,
                "geometry": parent_site.geometry,
                "orientation": _compute_orientation(parent_site.geometry, site.geometry),
                "installed_at": parent_site_id,
                "produced_by": f"Manufacturer:{manufacturer}",
                "connected_with": f"MicrowaveAntenna:{name_a}"
            },
        ]
    return GeoDataFrame(data, crs=CRS_ETRS89_UTM33N).set_index("id")


def load_cells_dataframe(data_dir: str, gdf_antennas: GeoDataFrame, df_services: DataFrame) -> GeoDataFrame:
    df = read_csv(os.path.join(data_dir, "cells.csv"), sep=";")
    df["served_by"] = "MobileAntenna:" + df["antenna_id"]
    df["service"] = "Service:" + df["service"]

    df = df.join(gdf_antennas, on="served_by", rsuffix="_a") \
        .join(df_services, on="service", rsuffix="_s")

    gdf = GeoDataFrame({
        "id": "Cell:" + df.cell_id,
        "name": df.cell_id,
        "antenna_operated_by": df.operated_by,
        "antenna_geometry": df.geometry,
        "antenna_installed_at": df.installed_at,
        "site_construction": df.site_construction,
        "service": df.service,
        "served_by": df.served_by,
        "service_frequency": df.frequency,
    }, crs=CRS_ETRS89_UTM33N, geometry="antenna_geometry")
    return gdf.set_index("id")


def load_manufacturers_dataframe() -> DataFrame:
    df = DataFrame([
        (ManufacturerEnum.HUAWEI,      "China",   35.1),
        (ManufacturerEnum.COMMSCOPE,   "USA",     13.3),
        (ManufacturerEnum.KATHREIN,    "Germany", 12.8),
        (ManufacturerEnum.ROSENBERGER, "Germany",  8.5),
        (ManufacturerEnum.AMPHENOL,    "USA",      6.1)
    ], columns=("name", "country", "market_share"))
    df["id"] = "Manufacturer:" + df["name"]
    return df.set_index("id")


def load_services_dataframe() -> DataFrame:
    df = DataFrame([
        (RadioAccessTechnologyEnum.NR,   700, BandEnum.N28   , 0.10, 2500),
        (RadioAccessTechnologyEnum.LTE,  800, BandEnum.B20   , 0.10, 2150),
        (RadioAccessTechnologyEnum.LTE,  900, BandEnum.B8    , 0.10, 2000),
        (RadioAccessTechnologyEnum.GSM,  900, BandEnum.GSM900, 0.07, 2000),
        (RadioAccessTechnologyEnum.LTE, 1500, BandEnum.B32   , 0.08, 1100),
        (RadioAccessTechnologyEnum.LTE, 1800, BandEnum.B3    , 0.18,  900),
        (RadioAccessTechnologyEnum.LTE, 2100, BandEnum.B1    , 0.10,  800),
        (RadioAccessTechnologyEnum.NR,  2100, BandEnum.N1    , 0.10,  800),
        (RadioAccessTechnologyEnum.LTE, 2600, BandEnum.B7    , 0.07,  600),
        (RadioAccessTechnologyEnum.NR,  3500, BandEnum.N78   , 0.10,  500),
    ], columns=("rat", "frequency", "band", "usage", "range"))
    # Add frequency-specific legal radiation limit (see https://www.gesetze-im-internet.de/bimschv_26/anhang_1.html)
    df["legal_radiation_limit"] = np.where(df.frequency < 2000, 1.375 * 0.0037 * df.frequency, 61 * 0.16)
    df["id"] = "Service:" + df["band"]
    return df.set_index("id")


def load_operators_dataframe() -> DataFrame:
    df = DataFrame([
        OperatorEnum.EINS_UND_EINS,
        OperatorEnum.TELEFONICA,
        OperatorEnum.TELEKOM,
        OperatorEnum.VODAFONE
    ], columns=["name"])
    df["id"] = "Operator:" + df["name"]
    return df.set_index("id")


def load_tiles_dataframe(gdf_regions: GeoDataFrame, tile_size: int = 100) -> GeoDataFrame:
    # Compute bounding box that contains all regions
    bbox = transform_bbox(gdf_regions.geometry.union_all().bounds, gdf_regions.crs.srs, CRS_ETRS89_UTM33N)
    # Generate equally spaced grid points
    xrange = np.arange(np.floor(bbox[0] / tile_size), np.ceil(bbox[2] / tile_size))
    yrange = np.arange(np.floor(bbox[1] / tile_size), np.ceil(bbox[3] / tile_size))
    data = []
    for x, y in itertools.product(xrange, yrange):
        tile_geometry = box(x * tile_size, y * tile_size, (x + 1) * tile_size, (y + 1) * tile_size)
        located_in = find_largest_intersection(gdf_regions, tile_geometry)
        if located_in is None:
            continue
        tile_name = f"{tile_size}mN{y:.0f}E{x:.0f}"
        population_density = gdf_regions.loc[located_in].population_density
        data.append((f"Tile:{tile_name}", tile_name, tile_geometry, located_in, population_density))
    
    gdf = GeoDataFrame(
        data,
        columns=("id", "name", "geometry", "located_in", "region_population_density"),
        crs=CRS_ETRS89_UTM33N
    )    
    return gdf.set_index("id")
   

def load_buildings_dataframe(data_dir: str) -> GeoDataFrame:
    with open(os.path.join(data_dir, "buildings.json"), "r", encoding="utf-8") as f:
        geojson = json.load(f)
    return GeoDataFrame.from_features(geojson, crs=CRS_ETRS89_UTM33N)


def load_pois_dataframe(data_dir: str) -> GeoDataFrame:
    with open(os.path.join(data_dir, "pois.json"), "r", encoding="utf-8") as f:
        geojson = json.load(f)
    gdf = GeoDataFrame.from_features(geojson, crs=CRS_ETRS89_UTM33N)
    gdf["id"] = "POI:" + gdf["name"]
    return gdf.set_index("id")
