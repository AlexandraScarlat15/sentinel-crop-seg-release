"""
Mask Data Loader
================

Abstract definition of a ground truth mask provider.

This module defines the contract for generating pixel-wise masks
aligned with Sentinel imagery, regardless of the original data source.

Examples of implementations:
- EuroCrops vector data
- LPIS / national datasets
- Raster ground truth
- AI-generated pseudo-labels
"""

from typing import Dict, Tuple, Any
import numpy as np


class MaskDataLoader:
    """
    Defines the interface for ground truth mask generation.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the mask data loader.

        Parameters
        ----------
        config : dict
            Configuration dictionary containing mask-related parameters
            (e.g. class schema, paths, filters).
        """
        self.config = config

    def get_class_mapping(self) -> Dict[int, str]:
        """
        Return the mapping between class IDs and semantic labels.

        Returns
        -------
        dict
            Dictionary mapping class_id -> class_name.
        """
        raise NotImplementedError

    def generate_mask(
        self,
        window: Tuple[int, int, int, int],
        spatial_metadata: Dict[str, Any]
    ) -> np.ndarray:
        """
        Generate a ground truth mask aligned with a Sentinel image window.

        Parameters
        ----------
        window : tuple of int
            (col_offset, row_offset, width, height)
        spatial_metadata : dict
            Spatial metadata of the reference Sentinel image
            (CRS, transform, resolution).

        Returns
        -------
        np.ndarray
            2D mask array of shape (H, W), containing class IDs.
        """
        raise NotImplementedError
