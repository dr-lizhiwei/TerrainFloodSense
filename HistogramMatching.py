import gdal
import cv2
from osgeo import gdal
import numpy as np
from ConfigFunction import clip_or_pad_image


def scale_image(image, new_min=0, new_max=100):
    """
    Scale the image pixel values to a new range [new_min, new_max].

    Parameters:
    - image: np.ndarray, the input image (grayscale).
    - new_min: int, the new minimum value of the scaled image.
    - new_max: int, the new maximum value of the scaled image.

    Returns:
    - scaled_image: np.ndarray, the scaled image.
    """
    old_min, old_max = np.min(image), np.max(image)
    scaled_image = (image - old_min) * (new_max - new_min) / ((old_max - old_min) + new_min)
    return scaled_image


def interpolate_histogram(template, num_bins=2000):
    """
    Interpolate the histogram of the template image to have a specified number of bins.

    Parameters:8
    - template: np.ndarray, the template image (grayscale).
    - num_bins: int, the number of bins for the interpolated histogram.

    Returns:
    - interp_t_values: np.ndarray, the interpolated histogram values.
    """
    template = template[(template > 0) & (template < 90)]
    t_values, t_counts = np.unique(template, return_counts=True)

    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # Interpolate to get the desired number of bins
    interp_bins = np.linspace(0, 1, num_bins)
    interp_t_values = np.interp(interp_bins, t_quantiles, t_values)

    return interp_t_values


def histogram_matching(source, interp_t_values):
    """
    Adjust the pixel values of the source image such that its histogram matches that of the template image.

    Parameters:
    - source: np.ndarray, the source image (grayscale).
    - interp_t_values: np.ndarray, the interpolated histogram values of the template.

    Returns:
    - matched: np.ndarray, the transformed source image with a histogram matching the template.
    """
    oldshape = source.shape
    source_flat = source.ravel()

    mask = (source_flat > 0) & (source_flat < 90)
    source_valid = source_flat[mask]

    # Get the set of unique pixel values and their corresponding indices and counts
    s_values, bin_idx, s_counts = np.unique(source_valid, return_inverse=True, return_counts=True)

    # Calculate the cumulative distribution function (CDF) for the source image
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]

    # Interpolate to find the pixel values in the template image that correspond to the quantiles in the source image
    matched_values = np.interp(s_quantiles, np.linspace(0, 1, len(interp_t_values)), interp_t_values)

    # Create the matched image, keeping zero values和>75 values unchanged
    matched_image = source_flat.copy()
    matched_image[mask] = matched_values[bin_idx]

    return matched_image.reshape(oldshape)
def float_to_int_image(float_image):
    flattened = float_image.flatten()
    unique_values = np.unique(flattened)
    mapping = dict(zip(unique_values, range(len(unique_values))))
    int_image = np.empty_like(flattened, dtype=np.uint16)
    for i, value in enumerate(flattened):
        int_image[i] = mapping[value]
    int_image = int_image.reshape(float_image.shape)
    return int_image

def histogram_match(adjusted_image, WaterOccorg,length,classes):

    image_adj_occ = adjusted_image
    image_occ = WaterOccorg
    image_occ = np.where(image_occ == 128,0,image_occ)
    image_occ = clip_or_pad_image(image_occ, image_adj_occ.shape[1], image_adj_occ.shape[0])

    scaled_adj_occ = scale_image(image_adj_occ,new_min=np.min(image_occ),new_max=np.max(image_occ))
    interp_t_values = interpolate_histogram(image_occ, num_bins=classes)
    matched_image = histogram_matching(scaled_adj_occ, interp_t_values)

    occ_ajst = np.where(image_occ >= length, image_occ, matched_image)
    occ_org_ajst = (image_occ + occ_ajst) / 2
    occ_to_Int = float_to_int_image(occ_org_ajst)

    return occ_to_Int




