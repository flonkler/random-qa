from sklearn.neighbors import NearestNeighbors
from shapely.geometry import LineString
from collections import Counter
import numpy as np

from pandas import DataFrame
from geopandas import GeoDataFrame
import random
import itertools

from random_qa.graph.node import ConnectionEnum

MAX_DEPTH = 3
MIN_DISTANCE = 250
MAX_DISTANCE = 3000
BUDGETS = [2, 2, 1]

def compute_site_connection(gdf_sites: GeoDataFrame, mmwave_ratio: float = 0.2, seed: int = 0xe20074) -> GeoDataFrame:
    """TODO"""
    # Initialize a random number generator with a fixed seed
    rng = random.Random(seed)
    points = np.asarray([(point.x, point.y) for point in gdf_sites.geometry])
    nn_distances, nn_indices = NearestNeighbors(n_neighbors=10).fit(points).kneighbors(points)
    # Remove points that match exactly
    nn_distances = nn_distances[:, 1:]
    nn_indices = nn_indices[:, 1:]

    mmwave_candidates = list(filter(
        lambda i: all(nn_distances[i, :] > MIN_DISTANCE) and np.sum(nn_distances[i, :] <= MAX_DISTANCE) > 3,
        range(len(points))
    ))
    mmwave_nodes = set(rng.sample(mmwave_candidates, k=int(len(points) * mmwave_ratio)))
    fiber_nodes = set(range(len(points))) - mmwave_nodes
    mmwave_connections = {}
    node_counter = Counter()

    def _resolve_exisiting_connection(i: int | None) -> tuple[int, ...] | None:
        if i is None:
            return None
        if i in fiber_nodes:
            return (i,)
        parents = _resolve_exisiting_connection(mmwave_connections.get(i))
        return (i, *parents) if parents is not None else None

    def _find_connection(i: int, depth = 1) -> tuple[int, ...]:
        if depth == MAX_DEPTH:
            return

        for neighbor_idx, distance in zip(nn_indices[i, :], nn_distances[i, :]):
            # Stop if neighbor distance exceeds the maximum
            if distance > MAX_DISTANCE:
                break

            # Check if a connection to a fiber node already exists
            connection_candidate = _resolve_exisiting_connection(neighbor_idx) 
            # Otherwise search for a new connection
            if connection_candidate is None:
                connection_candidate = _find_connection(neighbor_idx, depth=depth + 1)
        
            # Unable to find a connection, continue with next neighboring node
            if connection_candidate is None:
                continue

            # Check if all nodes in the connection have enough budget
            if all(node_counter.get(idx, 0) < budget for idx, budget in zip(connection_candidate[::-1], BUDGETS)):
                return (i, *connection_candidate)
   
    data = []
    for i in range(len(gdf_sites)):
        if i in mmwave_nodes:
            if i not in mmwave_connections:
                connection = _find_connection(i)
                for child, parent in itertools.pairwise(connection):
                    node_counter.update([parent])
                    mmwave_connections[child] = parent
            data.append((ConnectionEnum.MICROWAVE, gdf_sites.index[mmwave_connections[i]]))
        else:
            data.append((ConnectionEnum.FIBER, None))

    df = DataFrame(data, columns=("connection", "connection_provided_by"), index=gdf_sites.index)
    return gdf_sites.join(df)


def compute_cell_coverage(
    gdf_cells: GeoDataFrame, gdf_tiles: GeoDataFrame, gdf_buildings: GeoDataFrame, df_services: DataFrame
) -> GeoDataFrame:
    data = []
    indoor_area = gdf_buildings.union_all()
    # Only consider tiles that intersect with a building for indoor coverage
    gdf_tiles_indoor = gdf_tiles[gdf_tiles.geometry.centroid.within(indoor_area)]
    # Remaining tiles are considered for outdoor coverage
    gdf_tiles_outdoor = gdf_tiles[~gdf_tiles.geometry.centroid.within(indoor_area)]
    gdf_cells_modified = gdf_cells.assign(is_indoor=gdf_cells.site_construction == "Indoor")
    for (is_indoor, service, _), group_gdf in gdf_cells_modified.groupby(["is_indoor", "service", "antenna_operated_by"]):
        # Switch between indoor or outdoor tiles
        gdf_tiles = gdf_tiles_indoor if is_indoor else gdf_tiles_outdoor
        # Compute nearest cell for each tile centroid
        antenna_points = np.asarray([(point.x, point.y) for point in group_gdf.antenna_geometry])
        tile_centroids = np.asarray([(point.x, point.y) for point in gdf_tiles.geometry.centroid])
        distances, indices = NearestNeighbors(n_neighbors=1).fit(antenna_points).kneighbors(tile_centroids)
        for tile, distance, neighbor_index in zip(gdf_tiles.iloc, distances[:, 0], indices[:, 0]):
            # Skip tiles that are too far away (threshold depends on the cell's service)
            if distance > df_services.loc[service].range:
                continue
            # Lookup cell and tile object from dataframes
            cell = group_gdf.iloc[neighbor_index]
            # Ensure that the antenna and tile are within the same area
            if is_indoor:
                is_same_building = (
                    gdf_buildings.contains(tile.geometry.centroid) & gdf_buildings.contains(cell.antenna_geometry)
                ).all()
                if not is_same_building:
                    continue
            # Compute the area of the tile in square kilometers
            tile_area = tile.geometry.area / 1e6
            # Estimate the user count based on the population density of the region the tile is located in
            estimated_usercount = tile.region_population_density * tile_area
            data.append((cell.name, tile.name, tile_area, estimated_usercount))

    df = DataFrame(data, columns=("cell_id", "tile_id", "area", "user_count")) \
        .groupby("cell_id").agg({"user_count": "sum", "area": "sum", "tile_id": list}) \
        .rename(columns={"tile_id": "covered_tiles"})
    gdf_cells = gdf_cells.join(df, how="left").fillna(0)
    gdf_cells.covered_tiles = gdf_cells.covered_tiles.replace(0, None)
    return gdf_cells
