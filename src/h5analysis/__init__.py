"""Library to analyse, plot, and export data stored in HDF5 files."""

try:
    from ._version import version as __version__
    from ._version import version_tuple
except ImportError:
    __version__ = "unknown version"
    version_tuple = (0, 0, "unknown version")

__author__ = 'Patrick Braun'
