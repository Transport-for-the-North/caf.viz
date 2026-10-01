"""Download (and cache) open GeoSpatial data for examples."""

##### IMPORTS #####

import dataclasses
import enum
import logging
import pathlib
from typing import TypedDict

import geopandas as gpd
import pooch

##### CONSTANTS #####

LOG = logging.getLogger(__name__)


class _Source(TypedDict):
    filename: str
    url: str
    hash: str | None
    attribution: str
    epsg: int | None


_SOURCES: dict[str, _Source] = {
    "uk_boundaries_topo": {
        "filename": "uk_boundaries_topo.json",
        "url": "https://raw.githubusercontent.com/ONSdigital/uk-topojson/refs/heads/main/output/topo.json",
        "hash": "9feaf90628fee8cf713821801c17fff47691cfa03b60262d4b68e355c485f23d",
        "attribution": "UK Boundaries (Office National Statistics),"
        " source: https://github.com/ONSdigital/uk-topojson",
        "epsg": 4326,
    }
}

_CACHE = pooch.create(
    pooch.os_cache(__package__),
    "",
    registry={value["filename"]: value["hash"] for value in _SOURCES.values()},
    urls={value["filename"]: value["url"] for value in _SOURCES.values()},
)

##### CLASSES & FUNCTIONS #####


def _get_source(name: str) -> _Source:
    normalised = name.strip().lower().replace(r" ", "_")
    if normalised not in _SOURCES:
        raise KeyError(f"no dataset with name: {name!r}")

    return _SOURCES[normalised]


def _get_path(name: str) -> pathlib.Path:
    source = _get_source(name)
    return pathlib.Path(_CACHE.fetch(source["filename"]))


@dataclasses.dataclass
class _FileInfo:
    filename: str
    layer: str | None = None
    columns: list[str] | None = None

    _source: _Source | None = None

    @property
    def source(self) -> _Source:
        """File source information."""
        if self._source is None:
            self._source = _get_source(self.filename)
        return self._source

    def get_path(self) -> pathlib.Path:
        """Download file to cache and return path."""
        return _get_path(self.filename)

    @property
    def attribution(self) -> str:
        """Attribution text for the source."""
        return self.source["attribution"]

    @property
    def epsg(self) -> int | None:
        """Geospatial EPSG for the data."""
        return self.source["epsg"]


class Datasets(enum.StrEnum):
    """Datasets which are available, will be downloaded and cached."""

    ONS_UK = enum.auto()
    ONS_COUNTRY = enum.auto()
    ONS_REGION = enum.auto()
    ONS_COUNTY = enum.auto()
    ONS_METRO = enum.auto()
    ONS_CA = enum.auto()
    ONS_UTLA = enum.auto()
    ONS_LTLA = enum.auto()

    def get_info(self) -> _FileInfo:
        """Get information for the source file."""
        return _FILE_INFO[self]


_FILE_INFO: dict[Datasets, _FileInfo] = {
    Datasets.ONS_UK: _FileInfo("uk_boundaries_topo", "uk"),
    Datasets.ONS_COUNTRY: _FileInfo("uk_boundaries_topo", "ctry"),
    Datasets.ONS_REGION: _FileInfo("uk_boundaries_topo", "rgn"),
    Datasets.ONS_COUNTY: _FileInfo("uk_boundaries_topo", "cty"),
    Datasets.ONS_METRO: _FileInfo("uk_boundaries_topo", "mcty"),
    Datasets.ONS_CA: _FileInfo("uk_boundaries_topo", "cauth"),
    Datasets.ONS_UTLA: _FileInfo("uk_boundaries_topo", "utla"),
    Datasets.ONS_LTLA: _FileInfo("uk_boundaries_topo", "ltla"),
}


def fetch_dataset(dataset: Datasets) -> tuple[gpd.GeoDataFrame, str]:
    """Download dataset to cache and read.

    Returns
    -------
    gpd.GeoDataFrame
        Example dataset.
    str
        Source attribution text.
    """
    info = dataset.get_info()

    data = gpd.read_file(
        info.get_path(), layer=info.layer, columns=info.columns, engine="pyogrio"
    )
    if info.epsg is not None:
        data = data.set_crs(epsg=info.epsg)
    return data, info.attribution
