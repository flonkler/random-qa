from geopandas import GeoDataFrame, GeoSeries
from pandas import DataFrame, concat
import itertools
import numpy as np
from shapely.geometry import Polygon, Point, box
from typing import Iterator

from random_qa.graph import schema, GraphNode, GraphRelation
from random_qa.graph.node import ConnectionEnum, ConstructionEnum
from random_qa.utils import transform_bbox, CRS_WGS84_GPS, CRS_ETRS89_UTM33N, find_largest_intersection


def _base_generator(data: DataFrame | GeoDataFrame, cls: object) -> Iterator[GraphNode]:
    for _id, row in data.iterrows():
        yield cls(id=_id, **row.to_dict())

def generate_regions(gdf_regions: GeoDataFrame) -> Iterator[GraphNode]:
    return _base_generator(gdf_regions, schema.Region)

def generate_services(df_services: DataFrame) -> Iterator[GraphNode]:
    return _base_generator(df_services, schema.Service)

def generate_manufacturers(df_manufacturers: DataFrame) -> Iterator[GraphNode]:
    return _base_generator(df_manufacturers, schema.Manufacturer)

def generate_operators(df_operators: DataFrame) -> Iterator[GraphNode]:
    return _base_generator(df_operators, schema.Operator)

def generate_pois(df_pois: DataFrame) -> Iterator[GraphNode]:
    return _base_generator(df_pois, schema.POI)

def generate_tiles(gdf_tiles: GeoDataFrame) -> Iterator[GraphNode | GraphRelation]:
    for tile_id, tile in gdf_tiles.iterrows():
        yield schema.Tile(id=tile_id, name=tile["name"], area=tile.geometry.area / 1e6)
        yield schema.LocatedInRelation(in_id=tile.located_in, out_id=tile_id)

def generate_pois(
    gdf_pois: GeoDataFrame, gdf_regions: GeoDataFrame, gdf_tiles: GeoDataFrame
) -> Iterator[GraphNode | GraphRelation]:
    for poi_id, poi in gdf_pois.iterrows():
        yield schema.POI(id=poi_id, name=poi["name"])
        located_in = find_largest_intersection(gdf_regions, poi.geometry)
        if located_in is not None:
            yield schema.LocatedInRelation(in_id=located_in, out_id=poi_id)
        
        for tile_id in gdf_tiles[gdf_tiles.intersects(poi.geometry)].index:
            yield schema.LocatedInRelation(in_id=tile_id, out_id=poi_id)


def generate_sites(
    gdf_sites: GeoDataFrame, gdf_regions: GeoDataFrame, gdf_pois: GeoDataFrame
) -> Iterator[GraphNode | GraphRelation]:
    for site_id, site in gdf_sites.iterrows():
        site_region = gdf_regions[gdf_regions.contains(site.geometry)]
        if len(site_region) == 0:
            # Skip site because it is not located in any region
            continue

        yield schema.Site(
            id=site_id,
            name=str(site["name"]),
            coordinates=site.coordinates,
            construction=site.construction,
            connection=site.connection
        )
        yield schema.LocatedInRelation(in_id=site_region.iloc[0].name, out_id=site_id)
        for operator in site.operated_by:
            yield schema.OperatedByRelation(in_id=operator, out_id=site_id)
        if site.connection_provided_by is not None:
            yield schema.ConnectionProvidedBy(in_id=site.connection_provided_by, out_id=site_id)
        
        for poi_id in gdf_pois[gdf_pois.distance(site.geometry) < 500].index:
            yield schema.NearByRelation(in_id=poi_id, out_id=site_id)


def generate_antennas(
    gdf_mobile_antennas: GeoDataFrame,
    gdf_microwave_antennas: GeoDataFrame
) -> Iterator[GraphNode | GraphRelation]:
    for antenna_id, antenna in concat((gdf_mobile_antennas, gdf_microwave_antennas)).iterrows():
        if antenna_id.startswith("MobileAntenna:"):
            yield schema.MobileAntenna(id=antenna_id, name=antenna["name"], orientation=antenna.orientation)
            yield schema.OperatedByRelation(in_id=antenna.operated_by, out_id=antenna_id)
        elif antenna_id.startswith("MicrowaveAntenna:"):
            yield schema.MicrowaveAntenna(id=antenna_id, name=antenna["name"], orientation=antenna.orientation)
            yield schema.ConnectedWithRelation(in_id=antenna.connected_with, out_id=antenna_id)
        else:
            raise ValueError(f"Malformed antenna ID {antenna_id!r}")
        
        yield schema.InstalledAtRelation(in_id=antenna.installed_at, out_id=antenna_id)
        yield schema.ProducedByRelation(in_id=antenna.produced_by, out_id=antenna_id)        


def generate_cells(gdf_cells: GeoDataFrame) -> Iterator[GraphNode | GraphRelation]:
    for cell_id, cell in gdf_cells.iterrows():
        yield schema.Cell(id=cell_id, name=cell["name"], user_count=cell.user_count, area=cell["area"])
        yield schema.AvailableInRelation(in_id=cell_id, out_id=cell.service)
        yield schema.ServedByRelation(in_id=cell.served_by, out_id=cell_id)
        if cell.covered_tiles is None:
            continue
        for tile_id in cell.covered_tiles:
            yield schema.CoveredByRelation(in_id=cell_id, out_id=tile_id)
