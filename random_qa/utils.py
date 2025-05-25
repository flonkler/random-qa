import itertools
import numpy as np
import pyproj
import re
import random
import tilemapbase
from inspect import cleandoc
from typing import Iterator, Any
from datetime import datetime, timedelta
from shapely.geometry import box, Polygon
from geopandas import GeoDataFrame
from matplotlib.axes import Axes


BBOX_LEIPZIG_WGS84 = box(12.2366519, 51.2381704, 12.5425373, 51.4481145)

TILE_PROVIDER = tilemapbase.tiles.Tiles(
    "https://sgx.geodatenzentrum.de/wmts_basemapde/tile/1.0.0/de_basemapde_web_raster_grau/default/GLOBAL_WEBMERCATOR/{zoom}/{y}/{x}.png",
    "Geodatenzentrum (grau)"
)

CRS_WEBMERCATOR   = "epsg:3857"
CRS_WGS84_GPS     = "epsg:4326"
CRS_ETRS89_UTM33N = "epsg:25833"

def transform_bbox(
    bbox: tuple[float, float, float, float], source_crs: str, target_crs: str, as_dict: bool = False
) -> tuple[float, float, float, float] | dict[str, float]:
    """Transform a bounding box from one coordinate reference system to another.
    
    Parameters:
        bbox: extent of the bounding box in the format `(xmin, ymin, xmax, ymax)`
        source_crs: coordinate reference system of the input
        target_crs: coordinate reference system of the output
        as_dict: If set to `True`, extent of the bounding box is returned as key-value pairs. The keys are dependent on
            the given target CRS. Otherwise the result is returned as a tuple (default).
    
    Returns: transformed bounding box as a tuple `(xmin, ymin, xmax, ymax)` or a dictionary
    """
    transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)
    corners = np.array(
        [transformer.transform(x, y) for x, y in itertools.product((bbox[0], bbox[2]), (bbox[1], bbox[3]))]
    )
    xmin, xmax = corners[:, 0].min(), corners[:, 0].max()
    ymin, ymax = corners[:, 1].min(), corners[:, 1].max()
    if as_dict:
        if target_crs.lower() == CRS_WGS84_GPS:
            return {"min_lon": xmin, "max_lon": xmax, "min_lat": ymin, "max_lat": ymax}
        else:
            return {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax}
    else:
        return (xmin, ymin, xmax, ymax)


def random_id_generator(start: int, stop: int, seed: int | None = None) -> Iterator[int]:
    """Creates a generator that yields a randomly chosen unique ID that is within a specified number range.
    
    Parameters:
        start: start of the number range (i.e., the smallest ID)
        stop: stop of the number range (i.e., all IDs are less than this value)
        seed: Optional seed for the random number generator
    
    Yields: numeric ID
    """
    rng = random.Random(seed)
    available_ids = list(range(start, stop))
    while len(available_ids) > 0:
        yield available_ids.pop(rng.randrange(0, len(available_ids)))


def sanitize_multiline_string(value: str, fold_newlines: bool = False) -> str:
    """Remove unnecessary indentation and whitespaces from a multiline string (e.g., __doc__)

    Parameters:
        value: (Multiline) string that should be cleaned
        fold_newlines: If `True`, linebreaks are replaced by single whitespaces. Otherwise, linebreaks are not modified.
    
    Returns: Cleaned input string
    """
    value = cleandoc(value)
    if fold_newlines:
        value = value.replace("\n", " ")
    return value.strip()


def extract_codeblock(value: str) -> str | None:
    """Extract a Markdown-styled codeblock from a string
    
    Parameters:
        value: Raw input string
        
    Returns: Content inside the codeblock or `None` if no codeblock was found
    """
    m = re.search(r"```.*?\n(.*?)(?:$|\n```)", value, flags=re.DOTALL)
    if m is None:
        return None

    return m.group(1).strip()


def plot_tilemap(ax: Axes, area_of_interest: Polygon | None = None) -> None:
    """TODO"""
    if area_of_interest is not None:
        bbox = transform_bbox(area_of_interest.bounds, CRS_WEBMERCATOR, CRS_WGS84_GPS)
    else:
        bbox = BBOX_LEIPZIG_WGS84.bounds
    
    extent = tilemapbase.Extent.from_lonlat(bbox[0], bbox[2], bbox[1], bbox[3]).to_project_3857()
    plotter = tilemapbase.Plotter(extent, TILE_PROVIDER, width=ax.bbox.width)
    plotter.plot(ax)


def find_largest_intersection(gdf: GeoDataFrame, geometry: Polygon) -> str | int | None:
    """TODO"""
    intersections = gdf.intersection(geometry).area
    if (intersections == 0).all():
        return None
    else:
        return intersections.idxmax()
