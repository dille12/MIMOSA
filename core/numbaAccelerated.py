from numba import jit
import numpy as np
import cv2
import scipy.ndimage as sc
@jit(nopython = True)
def thresholdArrayJIT(image_array, luminosity, threshold):
    # Convert to grayscale by averaging RGB values

    #im = np.ones(image_array.shape, dtype=np.uint16) * 255
    
    # Create a binary mask based on the threshold
    binary_mask = (luminosity > threshold).astype(np.uint8)  # 0 or 1

    return image_array * binary_mask[:, :, np.newaxis]


@jit(nopython=True)
def toleranceJIT(image_array, luminosity, threshold, tolerance):
    # Compute threshold range
    minv = threshold - tolerance
    maxv = threshold + tolerance

    # Create a binary mask using element-wise logical operations
    binary_mask = ((luminosity > minv) & (luminosity < maxv)).astype(np.uint8)

    # Apply mask to image (assuming image_array is in RGB format)
    return image_array * binary_mask[:, :, np.newaxis]


@jit(nopython = True)
def upperThresholdArrayJIT(image_array, luminosity, threshold):
    # Convert to grayscale by averaging RGB values

    #im = np.ones(image_array.shape, dtype=np.uint16) * 255
    
    # Create a binary mask based on the threshold
    binary_mask = (luminosity < threshold).astype(np.uint8)  # 0 or 1

    return image_array * binary_mask[:, :, np.newaxis]


@jit(nopython=True)
def sobel_edge_detection_jit(image_array):
    height, width = image_array.shape
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]])
    gradient_magnitude = np.zeros_like(image_array, dtype=np.float32)

    # Apply Sobel kernel manually
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # Apply kernels
            gx = np.sum(image_array[i-1:i+2, j-1:j+2] * sobel_x)
            gy = np.sum(image_array[i-1:i+2, j-1:j+2] * sobel_y)

            # Compute gradient magnitude
            gradient_magnitude[i, j] = np.sqrt(gx**2 + gy**2)

    # Normalize and return
    gradient_magnitude = (gradient_magnitude / gradient_magnitude.max() * 255).astype(np.uint8)
    return gradient_magnitude



def CannyDetection(image_array, greyscale_image, threshold):

    if image_array.dtype != np.uint8:
        image_array = (255 * (image_array / image_array.max())).astype(np.uint8)

    if greyscale_image.dtype != np.uint8:
        greyscale_image = (255 * (greyscale_image / greyscale_image.max())).astype(np.uint8)

    binary_mask = cv2.Canny(greyscale_image, threshold1=threshold, threshold2=150)
    return image_array * binary_mask[:, :, np.newaxis]

@jit(nopython=True)
def optimized_island_fill(image, labeled_array, num_features, max_size=10):
    # Create histogram of labels to count pixels in each region in one pass
    # Add 1 for background (label 0)
    region_sizes = np.zeros(num_features + 1, dtype=np.int32)
    
    # Use flatten for faster iteration
    flat_labels = labeled_array.ravel()
    for label in flat_labels:
        region_sizes[label] += 1
    
    # Create output image
    output_image = image.copy()
    
    # Create mask of regions to fill (regions smaller than max_size)
    small_regions = np.zeros(num_features + 1, dtype=np.uint8)
    # Skip label 0 (background)
    for label in range(1, num_features + 1):
        if region_sizes[label] < max_size:
            small_regions[label] = 1
    
    # Apply mask in one operation
    flat_output = output_image.ravel()
    for i, label in enumerate(flat_labels):
        if small_regions[label]:
            flat_output[i] = 255
    
    return output_image

@jit(nopython=True)
def AcceleratedIslandRemoval(labeled_array, num_features, image, output_image, max_size):
    # Precompute dimensions
    rows, cols = labeled_array.shape
    is_rgb = image.ndim == 3

    for label in range(1, num_features + 1):
        region_size = 0

        # First pass: Compute the region size
        for r in range(rows):
            for c in range(cols):
                if labeled_array[r, c] == label:
                    region_size += 1

        # Second pass: Set pixels in small regions to black
        if region_size < max_size:
            for r in range(rows):
                for c in range(cols):
                    if labeled_array[r, c] == label:
                        if is_rgb:
                            output_image[r, c, :] = 0
                        else:
                            output_image[r, c] = 0

    return output_image


@jit(nopython=True)
def filter_labels(image1, selected_labels):
    selected = np.zeros_like(image1, dtype=np.uint8)
    max_label = selected_labels.max()
    lookup = np.zeros(max_label + 1, dtype=np.uint8)
    for lbl in selected_labels:
        lookup[lbl] = 1

    for i in range(image1.shape[0]):
        for j in range(image1.shape[1]):
            lbl = image1[i, j]
            if lbl > 0 and lbl < lookup.size and lookup[lbl]:
                selected[i, j] = 255
    return selected