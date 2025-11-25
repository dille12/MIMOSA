import numpy as np
import cv2
import pygame
from typing import List, Tuple, TYPE_CHECKING
import core.numbaAccelerated
import scipy.ndimage as sc
import scipy.spatial as sc2
import os
if TYPE_CHECKING:
    from core.node import Node
    from main import App
import AI.ai_core
import testing
from core.FOV import FOV
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import skimage



def gaussian_noise_reduction(Node: "Node", image: np.ndarray) -> np.ndarray:
    """

    SLIDER:0:6:0:Sigma
    MENUTYPE:Filters
    HELP:Reduces noise by gaussian filter. Higher sigma values increase blurriness.
    """
    sigma = Node.utility[0].value
    # If the image is grayscale (2D)
    if image.ndim == 2:
        denoised_image = sc.gaussian_filter(image, sigma=sigma)

    # If the image is RGB (3D), apply the Gaussian filter to each channel separately
    elif image.ndim == 3:
        denoised_image = np.zeros_like(image)
        for channel in range(image.shape[2]):
            denoised_image[..., channel] = sc.gaussian_filter(image[..., channel], sigma=sigma)

    else:
        raise ValueError("Input image must be a 2D grayscale or 3D RGB image")

    return denoised_image


def sharpen_image(image: np.ndarray) -> np.ndarray:
    """

    MENUTYPE:Filters
    HELP:Sharpens the image.
    """
    sharpen_kernel = np.array([[0, -1, 0],
                                [-1, 5, -1],
                                [0, -1, 0]])

    if image.ndim == 2:
        sharpened_image = sc.convolve(image, sharpen_kernel, mode='reflect')

    elif image.ndim == 3:
        sharpened_image = np.zeros_like(image)
        for channel in range(image.shape[2]):
            sharpened_image[..., channel] = sc.convolve(image[..., channel], sharpen_kernel, mode='reflect')

    else:
        raise ValueError("Input image must be a 2D grayscale or 3D RGB image")

    return np.clip(sharpened_image, 0, 255).astype(image.dtype)

def mean(image_array: np.ndarray) -> np.ndarray:

    """

    MENUTYPE:Misc
    HELP:Returns a greyscale array of the image.
    """
   
    # Compute the mean across the spatial dimensions (height, width) for each channel
    mean_values = np.mean(image_array, axis=(2))  # Shape: (3,)
    
    # Create an output image with the same shape, where all pixels have the mean RGB values
    mean_image = np.stack([mean_values] * 3, axis=-1)
    
    return mean_image

def pipeLineRoot(App: "App") -> np.ndarray:
    """
    MENUTYPE:Essential
    HELP:Outputs the image in current pipeline.
    """
    im = App.pipelineImage
    if not im:
        im = App.image

    image_array = pygame.surfarray.array3d(im)   
    if image_array.dtype != np.uint8:
        image_array = (255 * (image_array / image_array.max())).astype(np.uint8)

    return image_array

def inputImage(App: "App") -> np.ndarray:
    """
    MENUTYPE:Essential
    HELP:Inputs the original image for further operations. Connect to export node to display the image.
    """
   
    image_array = pygame.surfarray.array3d(App.image)   
    if image_array.dtype != np.uint8:
        image_array = (255 * (image_array / image_array.max())).astype(np.uint8)

    if App.renderRect:
        p1, p2 = App.renderRect
        s = round(p2 - p1)
        p1 = round(p1)
        x, y = int(p1[0]), int(p1[1])
        w, h = int(s[0]), int(s[1])
        image_array = image_array[x : x+w, y : y+h, :]

    return image_array

def pipelineExport(App: "App", Node: "Node", image_array: np.ndarray) -> None:
    """
    MENUTYPE:Essential
    HELP:Outputs the result to the pipeline, and to the viewport.
    """
    if App.PIPELINERESULT:
        App.PIPELINERESULT.addResult(image_array)
    App.setBackground(pygame.surfarray.make_surface(image_array))

def export(App: "App", Node: "Node", image_array: np.ndarray) -> None:
    """

    MENUTYPE:Essential
    HELP:Exports the image to the viewport background. This node initiates the whole pipeline, so ensure that the node network is fully connected. The save dropdown saves this image to the CNN train directory, so keep at Skip unless otherwise desired. 
    DROPDOWN:Saving:Skip:Save
    """
    SEL = Node.utility[0].dropDownSelections[Node.utility[0].value]
    
    if App.renderRect:
        p1, p2 = App.renderRect
        s = round(p2 - p1)
        p1 = round(p1)

        x, y = int(p1[0]), int(p1[1])
        w, h = int(s[0]), int(s[1])

        IMTEMP = pygame.surfarray.array3d(App.image)  

        IMTEMP[x : x+w, y : y+h, :] = image_array
        App.setBackground(pygame.surfarray.make_surface(IMTEMP))

    else:
        App.setBackground(pygame.surfarray.make_surface(image_array))

    if SEL == "Save":
        pygame.image.save(App.imageApplied, f"AI/validate_images/{App.imageName}.png")
        pygame.image.save(App.image, f"AI/train_images/{App.imageName}.png")
        App.notify("Image saved to AI train dir.")

def adjust_contrast(image: np.ndarray, Node: "Node") -> np.ndarray:
    """

    SLIDER:0:2:1:Contrast
    MENUTYPE:Contrast
    HELP:Adjust contrast based on the parameter. Higher values increase contrast, lower values decrease it.
    """
    factor = Node.utility[0].value
    if not isinstance(factor, (float, int)):
        raise ValueError("Contrast factor must be a positive number.")
    
    if factor == 0:
        return image

    # Convert to float for computation
    image = image.astype(np.float32)

    # Find the mean pixel value
    mean = np.mean(image, axis=(0, 1)) if image.ndim == 3 else np.mean(image)

    # Adjust contrast
    adjusted_image = mean + factor * (image - mean)

    # Clip values to maintain valid image pixel range (0-255 for uint8)
    return np.clip(adjusted_image, 0, 255).astype(image.dtype)


def lowerThreshold(Node: "Node", image_array: np.ndarray) -> np.ndarray:
    """

    SLIDER:0:255:0:Threshold
    MENUTYPE:Thresholding
    HELP:Cuts out pixels based on their luminosity.
    """

    m = mean(image_array)[:, :, 0]

    imarr = core.numbaAccelerated.thresholdArrayJIT(image_array, m, Node.utility[0].value)
    return imarr

def upperThreshold(Node: "Node", image_array: np.ndarray) -> np.ndarray:
    """

    SLIDER:0:255:255:Threshold
    MENUTYPE:Thresholding
    HELP:Cuts out pixels based on their luminosity.
    """

    m = mean(image_array)[:, :, 0]

    imarr = core.numbaAccelerated.upperThresholdArrayJIT(image_array, m, Node.utility[0].value)
    return imarr


def toleranceThreshold(Node: "Node", image_array: np.ndarray) -> np.ndarray:
    """

    SLIDER:0:255:127:Pixel Value
    SLIDER:0:50:5:Tolerance
    MENUTYPE:Thresholding
    HELP:Thresholds based on tolerance.
    """

    m = mean(image_array)[:, :, 0]

    imarr = core.numbaAccelerated.toleranceJIT(image_array, m, Node.utility[0].value, Node.utility[1].value)
    return imarr


def math(Node: "Node", image_array: np.ndarray, image_array2: np.ndarray) -> np.ndarray:
    """
    DROPDOWN:Operation:Add:Minus:Multiply:Min:Max
    MENUTYPE:Math
    HELP:Does matrix operations between two arrays, including addition, subtraction, multiplication, minimum, and maximum.
    """
    
    SEL = Node.utility[0].dropDownSelections[Node.utility[0].value]

    i1 = image_array.astype(np.float64) / 255
    i2 = image_array2.astype(np.float64) / 255
    
    if SEL == "Add":
        i3 = i1 + i2
    elif SEL == "Minus":
        i3 = i1 - i2
    elif SEL == "Multiply":
        i3 = i1 * i2
    elif SEL == "Min":
        i3 = np.minimum(i1, i2)
    elif SEL == "Max":
        i3 = np.maximum(i1, i2)
    else:
        raise ValueError("Invalid operation selected")

    i3 = np.clip(i3, 0, 1)
    i3 *= 255
    i3 = i3.astype(np.uint8)
    
    return i3

def fill_small_islands(image: np.ndarray, Node: "Node") -> np.ndarray:
    """
    SLIDER:0:100:0:Max Size
    MENUTYPE:Algorithm
    HELP:Fills black sections surrounded by populated pixels. Useful after thresholding to add back inner particles.
    """
    max_size = Node.utility[0].value ** 2
    if not max_size:
        return image
    
    # Work with just the first channel
    image_channel = image[:, :, 0].copy()
    
    # Create binary image and label connected components
    binary_image = (image_channel <= 5)
    labeled_array, num_features = sc.label(binary_image)
    
    # Get optimized output
    output_image = core.numbaAccelerated.optimized_island_fill(image_channel, labeled_array, num_features, max_size)
    
    # Stack the result into 3 channels
    return np.stack([output_image] * 3, axis=-1)
def image_to_binary(image: np.ndarray) -> np.ndarray:
    """

    MENUTYPE:Math
    HELP:Outputs a binary array.
    """
    
    threshold = 3
    if image.ndim == 3:  # Convert RGB to grayscale
        image = np.mean(image, axis=2).astype(np.uint8)

    # Create binary map based on the threshold
    binary_map = (image >= threshold).astype(np.uint8) * 255

    return np.stack([binary_map] * 3, axis=-1)

def edgeDetection(image_array: np.ndarray, Node: "Node") -> np.ndarray:
    """

    SLIDER:0:255:0:Threshold
    MENUTYPE:Algorithm
    HELP:Detects edges and outputs an binary array.
    """
    val = Node.utility[0].value
    return core.numbaAccelerated.CannyDetection(image_array, np.mean(image_array, axis=2), val)


def remove_small_islands(image: np.ndarray, Node: "Node") -> np.ndarray:
    """

    SLIDER:0:50:0:Max size
    MENUTYPE:Algorithm
    HELP:Removes pixel groups based on the maximum size parameter. Helpful to remove noise after thresholding.
    """
    if image.ndim not in [2, 3]:
        raise ValueError("Input image must be a 2D grayscale or 3D RGB image.")
    
    # Convert RGB image to grayscale if necessary
    if image.ndim == 3:
        grayscale_image = np.mean(image, axis=2).astype(np.uint8)
    else:
        grayscale_image = image

    max_size = int(Node.utility[0].value)
    if max_size == 0:
        return image

    # Create a binary mask: non-black pixels as 1, black pixels as 0
    binary_mask = grayscale_image > 0

    # Label connected components of non-black pixels
    labeled_array, num_features = sc.label(binary_mask)

    # Create a copy of the image to modify
    output_image = image.copy()

    output_image = core.numbaAccelerated.AcceleratedIslandRemoval(labeled_array, 
                                                                  num_features, 
                                                                  image, 
                                                                  output_image, 
                                                                  max_size)

    return output_image


def colorize_particles(image: np.ndarray) -> np.ndarray:
    """

    MENUTYPE:Essential
    HELP:Colorizes each distinct particle to a random color.
    """

    if image.ndim == 3:  # Convert RGB to grayscale
        image = np.mean(image, axis=2).astype(np.uint8)

    # Threshold to identify particles if needed (optional, e.g., for noisy images)
    threshold = 5  # Slider determines threshold for binarization
    binary_image = image > threshold  # Convert image to binary mask

    # Label connected components
    labeled_array, num_features = sc.label(binary_image)

    # Create an output image with the same dimensions but 3 color channels
    output_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    # Assign random colors to each particle
    np.random.seed(42)  # For consistent random colors during development
    colors = np.random.randint(0, 256, size=(num_features + 1, 3), dtype=np.uint8)

    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            label = labeled_array[r, c]
            if label > 0:  # Ignore background (label 0)
                output_image[r, c, :] = colors[label]

    return output_image


def mix(im1: np.ndarray, im2: np.ndarray, Node: "Node") -> np.ndarray:
    """

    SLIDER:0:1:0.5:Factor
    MENUTYPE:Math
    HELP:Mixes two arrays together based on the factor.
    """
    
    VAL = 1 - Node.utility[0].value

    i1 = im1.astype(np.float64)
    i2 = im2.astype(np.float64)

    i3 = i1 * VAL + i2 * (1-VAL)
    return i3.astype(np.uint16)




def particleSeparation(imageRaw: np.ndarray, Node: "Node") -> np.ndarray:
    """

    MENUTYPE:Algorithm
    HELP:Utilizes the watershed algorithm to segment overlapping particles. Adjust footprint to control the sensitivity.
    SLIDER:1:50:3:Footprint
    """

    #SEL = Node.utility[0].dropDownSelections[Node.utility[0].value]

    VAL = int(Node.utility[0].value)

    image = imageRaw[:, :, 0]

    distance = ndi.distance_transform_edt(image)
    coords = skimage.feature.peak_local_max(distance, footprint=np.ones((VAL, VAL)), labels=image)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = skimage.segmentation.watershed(-distance, markers, mask=image)
    


    # Find boundaries of segments
    boundaries = skimage.segmentation.find_boundaries(labels, mode='thick')

    # Create an output image and draw boundaries
    segmented_image = imageRaw.copy()
    segmented_image[boundaries] = [0, 0, 0]  # Set boundary pixels to black

    return segmented_image


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """
    MENUTYPE:Contrast
    HELP:Enhances the contrast automatically using local detection.
    """
    if image.ndim == 3:  # RGB
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced_image = cv2.merge((l, a, b))
        return cv2.cvtColor(enhanced_image, cv2.COLOR_LAB2RGB)
    else:  # Grayscale
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)
    

def cropY(image: np.ndarray, Node: "Node") -> np.ndarray:
    """
    SLIDER:0.1:1:1:Y
    MENUTYPE:Essential
    HELP:Crops the Y axis from the bottom.
    """
    factor = Node.utility[0].value
    Y = int(image.shape[1] * factor)
    return image[:, :Y, :]


def calculate_circularity(image: np.ndarray) -> dict:
    """
    MENUTYPE:Algorithm
    HELP:Calculates circularity and centroid for each particle.
    """

    # Ensure the image is RGB
    if image.ndim != 3:
        raise ValueError("Input image must be a 3D RGB image.")

    # Convert the image to grayscale to simplify processing
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Get unique labels (exclude black, which is the background)
    unique_labels = np.unique(grayscale)
    unique_labels = unique_labels[unique_labels > 0]  # Exclude 0 (background)

    circularity_results = {}

    for label in unique_labels:
        # Create a binary mask for the current particle
        particle_mask = (grayscale == label).astype(np.uint8)

        # Calculate the area (number of pixels in the mask)
        area = np.sum(particle_mask)

        # Find contours to compute the perimeter
        contours, _ = cv2.findContours(particle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = cv2.arcLength(contours[0], closed=True)

        # Calculate circularity
        if perimeter > 0:  # Avoid division by zero
            circularity = (4 * np.pi * area) / (perimeter ** 2)
        else:
            circularity = 0  # Edge case for single-pixel particles

        # Calculate the centroid using moments
        M = cv2.moments(particle_mask)
        if M["m00"] != 0:  # Avoid division by zero for centroid calculation
            centroid_x = int(M["m10"] / M["m00"])
            centroid_y = int(M["m01"] / M["m00"])
        else:
            centroid_x, centroid_y = -1, -1  # Invalid centroid for zero-area particles

        # Store the result for the current label
        circularity_results[label] = {
            "circularity": circularity,
            "centroid": (centroid_x, centroid_y)
        }

    return circularity_results


def exportText(item) -> None:
    """
    MENUTYPE:Essential
    HELP:Outputs a file of the raw data provided.
    """
    with open("output.txt", "w", encoding="utf-8", errors="ignore") as f:
        f.write(str(item))


def fatten_particles(image: np.ndarray, Node: "Node") -> np.ndarray:
    """
    SLIDER:0:5:0:Iterations
    HELP:Fatten the particles in an image by iteratively expanding particle boundaries.
    MENUTYPE:Algorithm
    """
    if image.ndim != 2:
        image = image[:, :, 0]
    iterations = int(Node.utility[0].value)
    # Define a convolution kernel to count neighbors
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    
    # Work on a copy of the image to preserve the original
    output_image = image.copy()

    for _ in range(iterations):
        # Create a binary mask of the current particles (1 for non-black pixels, 0 for black)
        binary_mask = (output_image > 0).astype(np.uint8)

        # Count the number of non-black neighbors for each pixel using convolution
        neighbor_count = sc.convolve(binary_mask, kernel, mode='constant', cval=0)
        
        # Find pixels to "fill" (black pixels with at least 3 non-black neighbors)
        to_fill = (neighbor_count >= 3) & (output_image == 0)
        
        # Update the image by filling the new pixels with the mode color of their neighbors
        if np.any(to_fill):
            # Create a padded version of the image for neighbor analysis
            padded_image = np.pad(output_image, pad_width=1, mode='constant', constant_values=0)

            # Assign the most frequent non-black color from neighbors to the filled pixels
            for i, j in zip(*np.where(to_fill)):
                i_pad, j_pad = i + 1, j + 1  # Adjust indices for the padded image
                neighborhood = padded_image[i_pad-1:i_pad+2, j_pad-1:j_pad+2].flatten()
                unique_colors, counts = np.unique(neighborhood[neighborhood > 0], return_counts=True)
                
                if len(unique_colors) > 0:
                    # Assign the most frequent neighbor color to the pixel
                    output_image[i, j] = unique_colors[np.argmax(counts)]

    return np.stack([output_image] * 3, axis=-1)


def invert(image: np.ndarray) -> np.ndarray:
    """
    HELP:Outputs an inverted image of the input
    MENUTYPE:Math
    """
    return 255 - image


def median_filter(image: np.ndarray, Node: "Node") -> np.ndarray:
    """
    SLIDER:1:5:3:Kernel Size
    HELP:Reduce noise using a median filter, replacing each pixel with the median of its neighbors.
    MENUTYPE:Filters
    """

    kernel_size = int(Node.utility[0].value)  # Get kernel size from slider

    # Apply the median filter to each channel separately
    output_image = np.zeros_like(image)
    for channel in range(3):
        output_image[..., channel] = sc.median_filter(image[..., channel], size=kernel_size)

    return output_image


def bilateral_noise_reduction(image: np.ndarray, Node: "Node") -> np.ndarray:
    """
    SLIDER:5:100:20:Sigma Space
    HELP:Reduce noise using a bilateral filter, preserving edges and smoothing flat regions.
    MENUTYPE:Filters
    """

    sigma_space = int(Node.utility[0].value)  # Spatial sigma (from slider 1)
    sigma_color = 20  # Color sigma (from slider 2)

    # Convert image to uint8 if not already
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    # Apply bilateral filter
    output_image = cv2.bilateralFilter(image, d=-1, sigmaColor=sigma_color, sigmaSpace=sigma_space)

    return output_image

def drawContours(image: np.ndarray) -> np.ndarray:
    """
    HELP:Draw contours around each particle.
    MENUTYPE:Algorithm
    """

    grayscale = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2GRAY)

    contours, hierarchy = cv2.findContours(grayscale, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    output = np.zeros(image.shape)

    im2 = cv2.drawContours(output, contours, -1, (255,255,255), 1)


    return im2


import cv2
import numpy as np

def circularity_filter(image: np.ndarray, Node: "Node") -> np.ndarray:
    """
    SLIDER:0:0.3:0.1:Circularity Threshold
    HELP:Filter non-circular objects in the image based on circularity. Circularity is defined as (4 * pi * area) / (perimeter^2). Non-circular regions are set to black while preserving pixel values.
    MENUTYPE:Filters
    """

    circularity_threshold = float(Node.utility[0].value)  # Circularity threshold (from slider)

    # Convert the grayscale 3D array to a 2D array for processing
    gray_2d = image[:, :, 0]

    # Ensure the image is binary (non-zero values to 255) for contour detection
    _, binary = cv2.threshold(gray_2d, 1, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank output image (same shape as input) initialized to black
    output_image = np.zeros_like(image)

    for contour in contours:
        # Calculate area and perimeter of the contour
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, closed=True)

        # Avoid division by zero and compute circularity
        if perimeter > 0:
            circularity = (4 * np.pi * area) / (perimeter ** 2)
        else:
            circularity = 0

        # If circularity exceeds the threshold, keep the original pixel values for this region
        if circularity >= circularity_threshold:
            # Create a mask for the current contour
            contour_mask = np.zeros_like(gray_2d, dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)

            # Use the mask to copy pixel values to the output image
            output_image[contour_mask == 255] = image[contour_mask == 255]

    return output_image


def elongation_filter(image: np.ndarray, Node: "Node") -> np.ndarray:
    """
    SLIDER:0:100:0.1:Threshold
    HELP:Filter out elongated/strip-like particles based on aspect ratio and area characteristics.
    MENUTYPE:Filters
    """

    elongation_threshold = float(Node.utility[0].value)  # Threshold for filtering
    
    # Convert to 2D grayscale
    gray_2d = image[:, :, 0]
    
    # Ensure binary image for contour detection
    _, binary = cv2.threshold(gray_2d, 1, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a blank output image
    output_image = np.zeros_like(image)
    
    for contour in contours:
        # Calculate area
        area = cv2.contourArea(contour)
        
        # Get minimum bounding rectangle
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = box.astype(np.int32)  # Changed from np.int0 to np.int32
        
        # Calculate dimensions of the bounding box
        width = rect[1][0]
        height = rect[1][1]
        
        # Calculate aspect ratio (longer side / shorter side)
        aspect_ratio = max(width, height) / min(width, height)
        
        # Calculate area to max bounding box length ratio
        max_bbox_length = max(width, height)
        area_to_length_ratio = area / max_bbox_length if max_bbox_length > 0 else 0
        
        # Filtering criteria
        # Lower aspect ratio means more circular
        # Higher area to length ratio indicates less strip-like
        if (area_to_length_ratio >= elongation_threshold):
            
            # Create a mask for the current contour
            contour_mask = np.zeros_like(gray_2d, dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
            
            # Copy pixel values to output image
            output_image[contour_mask == 255] = image[contour_mask == 255]
    
    return output_image


def drawSmoothedContours(image: np.ndarray) -> np.ndarray:
    # Convert to grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Morphological closing to remove small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(grayscale, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Approximate and smooth contours
    epsilon_factor = 0.01  # Adjust this to control smoothness
    approx_contours = [cv2.approxPolyDP(cnt, epsilon_factor * cv2.arcLength(cnt, True), True)
                       for cnt in contours]

    # Calculate convex hulls
    hulls = [cv2.convexHull(cnt) for cnt in approx_contours]

    # Draw the smoothed hulls
    output = np.zeros(image.shape)
    cv2.drawContours(output, hulls, -1, (255, 255, 255), 1)

    return output

def convexHulling(image: np.ndarray) -> np.ndarray:
    # Step 1: Convert to grayscale if the image is RGB
    if image.ndim == 3:  # Convert RGB to grayscale
        image = np.mean(image, axis=2).astype(np.uint8)

    # Step 2: Thresholding to identify particles
    threshold = 5  # Adjustable threshold
    binary_image = image > threshold  # Binary mask

    # Step 3: Label connected components
    labeled_array, num_features = sc.label(binary_image)

    # Step 4: Prepare output image (color version)
    output_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    # Step 5: Process each labeled particle
    for particle_idx in range(1, num_features + 1):  # 1 to num_features (skip background)
        # Extract pixel coordinates of the current particle
        particle_mask = labeled_array == particle_idx
        coordinates = np.column_stack(np.where(particle_mask))  # Y, X coordinates

        # Check if the particle has enough points
        if coordinates.shape[0] < 3:
            continue  # Skip small or insignificant particles

        # Step 6: Check if points span 2D space
        if np.all(coordinates[:, 1] == coordinates[0, 1]) or np.all(coordinates[:, 0] == coordinates[0, 0]):
            # Points are collinear (same x or y); draw a line instead
            min_point = coordinates[np.argmin(coordinates[:, 0])]
            max_point = coordinates[np.argmax(coordinates[:, 0])]
            p1 = tuple(min_point[::-1])  # X, Y
            p2 = tuple(max_point[::-1])
            cv2.line(output_image, p1, p2, (0, 0, 255), 1)  # Red line for collinear points
            continue

        # Step 7: Calculate the convex hull
        try:
            hull = sc2.ConvexHull(coordinates)
            hull_points = coordinates[hull.vertices]  # Vertices of the convex hull

            # Step 8: Draw the convex hull
            for i in range(len(hull_points)):
                p1 = tuple(hull_points[i][::-1])  # Swap Y, X -> X, Y for OpenCV
                p2 = tuple(hull_points[(i + 1) % len(hull_points)][::-1])
                cv2.line(output_image, p1, p2, (0, 255, 0), 1)  # Green lines

            # Optional: Draw the shortest distance on the convex hull
            shortest_distance = np.inf
            shortest_pair = None
            for i, point1 in enumerate(hull_points):
                for j, point2 in enumerate(hull_points):
                    if i != j:
                        dist = sc2.distance.euclidean(point1, point2)
                        if dist < shortest_distance:
                            shortest_distance = dist
                            shortest_pair = (point1, point2)

            # Draw the shortest line
            if shortest_pair:
                p1 = tuple(shortest_pair[0][::-1])
                p2 = tuple(shortest_pair[1][::-1])
                cv2.line(output_image, p1, p2, (255, 0, 0), 1)  # Blue for shortest line

        except Exception as e:
            print(f"Skipping particle {particle_idx}: {e}")

    return output_image


def KERAS_Model(App: "App", Node: "Node", image: np.ndarray) -> np.ndarray:
    """
    MENUTYPE:AI
    DROPDOWN:Stride:1:2/3:1/2:1/4
    DROPDOWN:Input size:256:384:512:1024
    DROPDOWN:3 Channel Data:True:False
    HELP:A neural network trained to detect lignin particles from a SEM image. Accuracy may vary.
    """

    SEL = Node.utility[0].dropDownSelections[Node.utility[0].value]
    SEL1 = Node.utility[1].dropDownSelections[Node.utility[1].value]
    SEL2 = Node.utility[2].dropDownSelections[Node.utility[2].value]
    rgb = SEL2 == "True"
    if SEL == "1/4":
        strideMod = 0.25
    elif SEL == "1/2":
        strideMod = 0.5
    elif SEL == "2/3":
        strideMod = 2/3
    else:
        strideMod = 1

    if SEL1 == "256":
        input_shape = [256,256]
    elif SEL1 == "384":
        input_shape = [384,384]
    elif SEL1 == "512":
        input_shape = [512,512]
    elif SEL1 == "1024":
        input_shape = [1024,1024]


    print("Initial input shape", image.shape)
    if image.ndim == 3:  # Convert RGB to grayscale
        image = np.mean(image.copy(), axis=2).astype(np.uint8)

    #input_shape = App.MODEL.input_shape[1:3]
    
    print("Input shape:", input_shape)
    
    stride = int(input_shape[0]*strideMod)
    print("STRIDE:", stride)

    result = AI.ai_core.predict_and_stitch(image, App.MODEL, window_size=input_shape, stride=stride, visualize = True, app = App, rgb=rgb)   
    print("Result shape:", result.shape, "Maximum", np.max(result))

    result = (255 * (result / np.max(result))).astype(np.uint8)

    return  np.stack([result] * 3, axis=-1)

def drawCircularityColoredParticles(image: np.ndarray, Node: "Node") -> np.ndarray:
    """
    HELP:Processes an image to segment particles, calculate circularity, and visualize them with brightness proportional to circularity (white for perfect circles, black for non-circular shapes).
    SLIDER:1:40:20:MinDistance
    """

    min_distance = int(Node.utility[0].value)

    # Convert to grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binary thresholding to create a binary image
    _, binary_image = cv2.threshold(grayscale, 127, 255, cv2.THRESH_BINARY)

    # Use the existing `separate_particles` function to label particles and get properties
    labeled_particles, particle_props = testing.separate_particles(binary_image, min_distance=min_distance)

    # Create an output image for visualization
    output = np.zeros_like(grayscale, dtype=np.uint8)

    # Map circularity to brightness (0: black, 1: white)
    for prop in particle_props:
        label = prop['label']
        circularity = prop['circularity']

        # Map circularity (0 to 1) to brightness (0 to 255)
        brightness = int(circularity * 255)

        # Colorize particles based on brightness
        output[labeled_particles == label] = brightness

    return output



def erode(image: np.ndarray, Node: "Node") -> np.ndarray:

    """
    HELP:Erodes particles based on the kernel size. 
    SLIDER:1:7:3:Kernel
    """
    # HELP defines the tool tip for the node.
    # SLIDER defines a adjustable slider, and the parameters are:
    # 1 - Minimum value
    # 7 - Maximum value
    # 3 - Initial value
    # Kernel - Name of the slider

    KERNEL = int(Node.utility[0].value) # The kernel size is defined from the slider value.

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL, KERNEL))  # An open-cv2 function is called to create a kernel based on the kernel size.

    erosionImage = cv2.erode(image, kernel, iterations=1) # The image is eroded using open-cv2 

    return erosionImage # The image is returned.


def dilate(image: np.ndarray, Node: "Node") -> np.ndarray:
    """
    HELP:Dilates particles based on the kernel size.
    SLIDER:1:7:3:Kernel
    """    

    KERNEL = int(Node.utility[0].value)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL, KERNEL))  # Adjust size as needed

    dilation = cv2.dilate(image, kernel, iterations=1)

    return dilation


def morphologicalOperation(image: np.ndarray, Node: "Node") -> np.ndarray:
    """
    MENUTYPE:Algorithm
    HELP:Provides a morphological operation based on the selected mode.
    DROPDOWN:Mode:erode:dilate:opening:closing:gradient
    SLIDER:1:7:3:Kernel
    """    

    KERNEL = int(Node.utility[1].value)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL, KERNEL))  # Adjust size as needed
    
    SEL = Node.utility[0].dropDownSelections[Node.utility[0].value]


    if SEL == "erode":
        im = cv2.erode(image, kernel, iterations=1)
    elif SEL == "dilate":
        im = cv2.dilate(image, kernel, iterations=1)
    elif SEL == "opening":
        im = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    elif SEL == "closing":
        im = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    elif SEL == "gradient":
        im = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)

    return im


def apply_CLAHE(image: np.ndarray) -> np.ndarray:
    """
    HELP:Apply Adaptive Histogram Equalization (CLAHE) to enhance contrast.
    MENUTYPE:Contrast
    """

    if image.ndim == 3:  # Convert RGB to grayscale
        image = np.mean(image.copy(), axis=2).astype(np.uint8)

    clip_limit=2.0
    tile_grid_size=(8, 8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    result = clahe.apply(image)
    return np.stack([result] * 3, axis=-1)


def filterPatchesByArea(image: np.ndarray, Node: "Node") -> Tuple[np.ndarray, np.ndarray]:
    """
    HELP:Detects particles and filters only those whose area falls within two slider values.
    MENUTYPE:Algorithm
    DROPDOWN:Selection:normal:invert
    SLIDER:0:2500:50:Min_Area
    SLIDER:10:2500:55:Max_Area
    """

    Node.utility[1].maxVal = Node.utility[2].value
    Node.utility[2].minVal = Node.utility[1].value + 1

    Node.utility[1].value = min(Node.utility[1].value, Node.utility[1].maxVal)
    Node.utility[2].value = max(Node.utility[2].value, Node.utility[2].minVal)

    MIN_AREA = int(Node.utility[1].value) ** 1.25  # Get Min Area from slider
    MAX_AREA = int(Node.utility[2].value) ** 1.25  # Get Max Area from slider
    SEL = Node.utility[0].dropDownSelections[Node.utility[0].value]
    inverted = SEL == "invert"

    if MIN_AREA >= MAX_AREA:
        return image

    # Ensure the image is binary (convert to grayscale if needed)
    if len(image.shape) == 3:  # If image is RGB or 3D, convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)  # Convert to binary mask

    # Find connected components
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask for filtered patches
    filtered_mask = np.zeros_like(binary)
    overflow_mask = np.zeros_like(binary)

    for contour in contours:
        area = cv2.contourArea(contour)

        b = MIN_AREA <= area <= MAX_AREA if not inverted else not (MIN_AREA <= area <= MAX_AREA)

        if b:
            cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)  # Keep the patch
        else:
            cv2.drawContours(overflow_mask, [contour], -1, 255, thickness=cv2.FILLED)  # Keep the patch

    # Apply the mask to the original image
    filtered_image = cv2.bitwise_and(image, image, mask=filtered_mask)

    overflow_image = cv2.bitwise_and(image, image, mask=overflow_mask)

    return filtered_image, overflow_image



def inputData(Node: "Node") -> dict:
    """
    HELP:Fetches image data from a text file.
    MENUTYPE:Data
    FILEINPUT:File
    """

    with open(Node.utility[0].value, encoding="utf-8", errors="ignore") as f:
        data = f.read()

    d = data.split("$")
    d2 = {}
    for x in d:
        x = x.rstrip("\n")
        y = x.split(" ")
        d2[y[0]] = y[1:]
    print(d2)
    return d2

def inputImageFile(Node: "Node") -> np.ndarray:
    """
    HELP:Loads an image file and outputs it as an array.
    MENUTYPE:Essential
    FILEINPUT:Image
    """

    path = Node.utility[0].value
    surf = pygame.image.load(path)
    image_array = pygame.surfarray.array3d(surf)

    if image_array.dtype != np.uint8:
        image_array = (255 * (image_array / image_array.max())).astype(np.uint8)

    return image_array



def getValueFromDict(d: dict, Node: "Node") -> list:
    """
    HELP:Gets a single value from a dictionary type object.
    MENUTYPE:Data
    DROPDOWN:Key:Default
    """

    Node.utility[0].dropDownSelections = list(d.keys())

    if Node.utility[0].value >= len(Node.utility[0].dropDownSelections):
        Node.utility[0].value = 0


    return d[Node.utility[0].dropDownSelections[Node.utility[0].value]]



def parseToFOV(l: list, image: np.ndarray) -> FOV:
    """
    HELP:Creates a field of view object to determine particle dimensions.
    MENUTYPE:Data
    """
    return FOV(l, image)


def processData(image: np.ndarray, FOV: FOV, App: "App") -> None:
    """
    HELP:Processes particles. Initializes a separate calculation, where the progress is seen in the topright of the software. After completion, particles can be inspected in Data Analysis.
    MENUTYPE:Data
    """
    # Step 1: Label connected particles
    labeled_image, num_features = sc.label(image)
    
    # Step 2: Measure particle areas in pixels
    particle_areas_px = np.bincount(labeled_image.ravel())[1:]  # Ignore background (label 0)

    # Step 3: Convert pixel areas to real-world areas
    pixel_area, unit = FOV.getAreaOfPixel()
    particle_areas_real = particle_areas_px * pixel_area

    # Step 4: Sort the areas for visualization
    sorted_areas = np.sort(particle_areas_real)


    #App.PROCESSEDDATA = labeled_image
    App.FOV = FOV

    App.startProcessing(labeled_image)

    print(f"Processed {num_features} particles.")
    print(f"Sorted real-world particle areas ({unit}²): {sorted_areas}")

    return


    num_bins = 25
    min_val = max(np.min(sorted_areas), 1e-6)  # Avoid log(0)
    max_val = np.max(sorted_areas)
    log_bins = np.logspace(np.log10(min_val), np.log10(max_val), num_bins)

    counts, bin_edges = np.histogram(sorted_areas, bins=log_bins)
      # Set x-axis to logarithmic

    # Plot the bar graph
    plt.figure(figsize=(10, 5))
    plt.bar(bin_edges[:-1], counts, width=np.diff(bin_edges), edgecolor='black', align='edge')
    plt.yscale('log')
    plt.xscale('log')

    # Labels and title
    plt.xlabel(f"Particle Area ({unit}²)")
    plt.ylabel("Number of Particles")
    plt.title("Particle Size Distribution")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Show plot
    plt.show()

def removeBorderTouching(Node: "Node", image_array: np.ndarray) -> np.ndarray:
    """
    MENUTYPE:Filters
    HELP:Removes labeled particles that touch the edges of the image (to exclude cut-off particles).
    """

    labeled = skimage.measure.label(image_array > 0, connectivity=2)
    if labeled.max() == 0:
        return np.zeros_like(image_array, dtype=np.uint8)

    # Find all labels touching borders
    border_labels = np.unique(np.concatenate([
        labeled[0, :],         # top row
        labeled[-1, :],        # bottom row
        labeled[:, 0],         # left column
        labeled[:, -1]         # right column
    ]))
    border_labels = border_labels[border_labels > 0]  # exclude background

    # Keep only labels not touching border
    mask = np.isin(labeled, border_labels, invert=True)
    im = image_array * mask
    out = (im.astype(np.uint8))
    return out

def mask(image: np.ndarray, Node: "Node", App: "App") -> np.ndarray:
    """
    HELP:Applies a mask to the image.
    MENUTYPE:Essential
    BUTTON:Set mask:configureMask
    """
    mask = Node.utility[0].value
    im = image.copy()
    if isinstance(mask, np.ndarray) and not App.editingMask:
        im *= mask[:, :, None]
        print("Masking")


    return im



def median_blur(image: np.ndarray, Node: "Node") -> np.ndarray:
    """
    SLIDER:1:15:3:Kernel Size
    MENUTYPE:Filters
    HELP:Reduces salt-and-pepper noise using a median filter. Higher kernel sizes remove more noise but may blur the image.
    """

    # Ensure odd kernel size
    ksize = int(Node.utility[0].value)
    if ksize % 2 == 0:
        ksize += 1

    # Apply median filter
    return cv2.medianBlur(image, ksize)


def particleIntersections(image_arr1: np.ndarray, image_arr2: np.ndarray, Node: "Node") -> np.ndarray:
    """
    DROPDOWN:Mode:Exclude:Include
    MENUTYPE:Math
    HELP:Takes two arrays as input, and takes either those particles that intersect, or those from the first input that don't intersect with any of the second input's particles.
    """
    mode = Node.utility[0].value  # 'Include' or 'Exclude'

    if image_arr1.ndim == 3:
        image1 = np.mean(image_arr1, axis=2).astype(np.uint8)
    else:
        image1 = image_arr1.copy()

    if image_arr2.ndim == 3:
        image2 = np.mean(image_arr2, axis=2).astype(np.uint8)
    else:
        image2 = image_arr2.copy()

    threshold = 5
    binary1 = image1 > threshold
    binary2 = image2 > threshold

    image1, _ = sc.label(binary1)
    image2, _ = sc.label(binary2)

    labels1 = np.unique(image1)
    labels1 = labels1[labels1 != 0]
    labels2 = np.unique(image2)
    labels2 = labels2[labels2 != 0]

    intersect_mask = (image1 != 0) & (image2 != 0)
    intersecting_labels = np.unique(image1[intersect_mask])

    if mode == "Include":
        selected_labels = intersecting_labels
    else:
        selected_labels = np.setdiff1d(labels1, intersecting_labels, assume_unique=True)

    out_mask = core.numbaAccelerated.filter_labels(image1, selected_labels)
    out_rgb = np.stack([out_mask] * 3, axis=-1)

    return out_rgb


def KERAS_SegmentParticles(App: "App", Node: "Node", image_arr1: np.ndarray) -> np.ndarray:
    """
    MENUTYPE:AI
    HELP:Runs per-particle segmentation using a U-Net model trained to separate overlapping particles. Each particle label is isolated, resized to 64x64, passed through the model, then resized back and inserted into the output mask.
    """

    # Prepare output mask

    if image_arr1.ndim == 3:
        image1 = np.mean(image_arr1, axis=2).astype(np.uint8)
    else:
        image1 = image_arr1.copy()

    threshold = 5
    binary1 = image1 > threshold

    labeled_image, _ = sc.label(binary1)

    output_mask = np.zeros_like(labeled_image, dtype=np.uint8)

    labels = np.unique(labeled_image)
    labels = labels[labels != 0]  # skip background

    for label in labels:
        particle_mask = (labeled_image == label).astype(np.uint8)
        if np.sum(particle_mask) < 256:
            output_mask[labeled_image == label] = 255  # keep small particles as is
            continue

        if "c" in App.keypress_held_down:
            print("Aborting...")
            App.notify("Aborting AI process...")
            App.AiProcessRect = None
            return None


        # Bounding box
        coords = np.argwhere(particle_mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1

        App.AiProcessRect = [y0, x0, y1, x1]


        # Crop particle and resize to 128x128
        WINDOW = 128
        crop = particle_mask[y0:y1, x0:x1]
        if crop.shape[0] < 2 or crop.shape[1] < 2:
            continue  # skip degenerate regions

        resized = skimage.transform.resize(crop, (WINDOW, WINDOW), order=1, preserve_range=True).astype(np.float32)
        input_tensor = resized[np.newaxis, ..., np.newaxis]  # shape (1, 128, 128, 1)

        # Predict mask
        prediction = App.MODELSEGM.predict(input_tensor, verbose=0)[0, ..., 0]
        
        #prediction = (prediction > 0.3).astype(np.uint8)

        resized = resized * (1 - prediction)  # Apply mask to resized image


        # Resize back to original crop size
        restored = skimage.transform.resize(resized, crop.shape, order=1, preserve_range=True)

        mask_in_bbox = particle_mask[y0:y1, x0:x1]
        restored_u8 = (restored * 255).astype(np.uint8)
        output_mask[y0:y1, x0:x1][mask_in_bbox == 1] = restored_u8[mask_in_bbox == 1]


    # Stack to RGB
    App.AiProcessRect = None
    return np.stack([output_mask] * 3, axis=-1)


def particleFiltration(Node: "Node", image_array: np.ndarray) -> np.ndarray:
    """
    MENUTYPE:Filters
    HELP:Filters labeled particles based on shape descriptors like eccentricity, aspect ratio, solidity, or feret diameter.
    DROPDOWN:Mode:Eccentricity:AspectRatio:Solidity:FeretDiameter
    SLIDER:0:100:50:Cutoff
    """

    MODE = Node.utility[0].dropDownSelections[Node.utility[0].value]
    CUTOFF = Node.utility[1].value / 100.0


    if image_array.ndim == 3:  # Convert RGB to grayscale
        image_array = np.mean(image_array, axis=2).astype(np.uint8)

    labeled = skimage.measure.label(image_array > 0)
    props = skimage.measure.regionprops(labeled)

    scores = np.zeros(len(props), dtype=np.float32)
    print(MODE)

    if MODE == "Eccentricity":
        for i, r in enumerate(props):
            scores[i] = r.eccentricity  # already in [0,1]

    elif MODE == "Solidity":
        for i, r in enumerate(props):
            scores[i] = r.solidity  # already in [0,1]

    elif MODE == "AspectRatio":
        for i, r in enumerate(props):
            if r.minor_axis_length > 0:
                scores[i] = r.major_axis_length / r.minor_axis_length
            else:
                scores[i] = 1.0  # minimal aspect ratio
        max_val = scores.max()
        if max_val > 1:
            scores = (scores - 1) / (max_val - 1)  # Normalize to [0,1]

    elif MODE == "FeretDiameter":
        feret_data = skimage.measure.regionprops_table(labeled, properties=("label", "feret_diameter_max"))
        scores = feret_data["feret_diameter_max"].astype(np.float32)
        max_val = scores.max()
        min_val = scores.min()
        print(max_val, min_val)
        if max_val > min_val:
            scores = (scores - min_val) / (max_val - min_val)
        else:
            scores[:] = 0.0

    keep_labels = [i + 1 for i, s in enumerate(scores) if s <= CUTOFF]
    out = np.isin(labeled, keep_labels).astype(np.uint8) * 255

    return np.stack([out] * 3, axis=-1)
