"""
Sentinel Data Loader
====================

Defines the interface for accessing Sentinel-2 imagery.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, List
import numpy as np


class SentinelDataLoader(ABC):
    """
    Abstract base class for Sentinel-2 data loaders.
    """

    @abstractmethod
    def get_spatial_metadata(self) -> Dict[str, any]:
        """
        Return spatial metadata of the Sentinel image.

        Expected keys:
        - width
        - height
        - crs
        - transform
        - resolution
        """
        pass

    @abstractmethod
    def read_window(
        self, window: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Read a spatial window from the Sentinel image.

        Parameters
        ----------
        window : tuple
            (col_offset, row_offset, width, height)

        Returns
        -------
        np.ndarray
            Array of shape (bands, height, width)
        """
        pass

    @abstractmethod
    def list_bands(self) -> List[str]:
        """
        List available spectral bands.
        """
        pass
