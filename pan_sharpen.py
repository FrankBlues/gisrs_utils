# -*- coding: utf-8 -*-
"""
PanSharpen method of remote sensing images.

Note:
    Only Brovey method here.

"""

import numpy as np


def calculate_ratio(rgb, pan, weight):
    """The ratio of the corresponding panchromatic pixel intensity
    to the sum of all the multispectral intensities with a weight
    value on the blue band used by the (Weighted) Brovey Pansharpen
    method.

    Args:
        rgb (list): List of multispectral image band array.
        pan (numpy ndarray): The panchromatic band array.
        weight (float): Weight of the blue band.

    Returns:
        numpy ndarray: The ratio array.

    """
    return pan / ((rgb[0] + rgb[1] + rgb[2] * weight) / (2 + weight))


def Brovey(rgb, pan, weight, pan_dtype):
    """Brovey Method: Each resampled, multispectral pixel is multiplied by
    the ratio of the corresponding panchromatic pixel intensity to the sum
    of all the multispectral intensities.

    Args:
        rgb (list): List of multispectral image band array.
        pan (numpy ndarray): The panchromatic band array.
        weight (float): Weight of the blue band.
        pan_dtype (numpy ndtype): Data type of the band data.

    Returns:
        pansharpened image data and the ratio.

    """
    with np.errstate(invalid='ignore', divide='ignore'):
        ratio = calculate_ratio(rgb, pan, weight)
    with np.errstate(invalid='ignore'):
        sharp = np.clip(ratio * rgb, 0, np.iinfo(pan_dtype).max)
        return sharp.astype(pan_dtype), ratio
