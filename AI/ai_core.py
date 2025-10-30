import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt



def sliding_window(image, window_size=(128, 128), stride=64, pad_mode='constant', sobel = False):
    """
    Generates sliding window patches from an image with edge padding.

    Args:
        image: np.ndarray, input image (2D or 3D).
        window_size: tuple, size of each window (height, width).
        stride: int, step size for sliding window.
        pad_mode: str, padding mode for np.pad.

    Returns:
        patches: np.ndarray of patches
        coordinates: np.ndarray of (y, x) coordinates for each patch
    """
    h, w = image.shape[:2]
    pad_h = (window_size[0] - h % stride) % window_size[0]
    pad_w = (window_size[1] - w % stride) % window_size[1]
    
    # Pad the image
    if len(image.shape) == 3:
        padded_image = np.pad(image, 
                              ((0, pad_h), (0, pad_w), (0, 0)), 
                              mode=pad_mode, constant_values=0)
    else:
        padded_image = np.pad(image, 
                              ((0, pad_h), (0, pad_w)), 
                              mode=pad_mode, constant_values=0)
    
    patches = []
    coordinates = []
    # Slide the window
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            x = min(x, w-window_size[1])
            y = min(y, h-window_size[0])
            patch = padded_image[y:y + window_size[0], x:x + window_size[1]]
            
            patches.append(patch)
            coordinates.append((y, x))
            

  
    
    return np.array(patches), np.array(coordinates), padded_image


import cv2

def genContrastEdges(img, kernel_size=3, contrast_threshold=0.2):
    """
    Computes edges based on local contrast differences.

    Args:
        img: np.ndarray, input grayscale image.
        kernel_size: int, size of the neighborhood window.
        contrast_threshold: float, threshold (0-1) to keep strong edges.

    Returns:
        edges: np.ndarray, contrast-based edge image.
    """
    img_gray = (img * 255).astype(np.uint8)  # Convert to uint8
    imgy, imgx = img_gray.shape[:2]
    contrast = np.zeros(img_gray.shape)

    for y in range(imgy):
        for x in range(imgx):
            miny = max(y - kernel_size, 0)
            maxy = min(y + kernel_size, imgy)
            minx = max(x - kernel_size, 0)
            maxx = min(x + kernel_size, imgx)
            window = img_gray[miny:maxy, minx:maxx]
            contrast[y, x, :] = np.max(window) - np.min(window)
    
    contrast -= np.min(contrast)
    contrast /= np.max(contrast)

    contrast[contrast > 0.75] = 0
    #contrast[contrast < 0.2] = 0 
    #contrast[contrast > 0] = 1

    #contrast = np.expand_dims(contrast, axis=-1)

    return contrast

def genContrastEdges2(img, kernel_size=3):
    """
    Computes edges based on local contrast differences using vectorized operations.

    Args:
        img: np.ndarray, input grayscale image.
        kernel_size: int, size of the neighborhood window.

    Returns:
        edges: np.ndarray, contrast-based edge image.
    """
    img_gray = (img * 255).astype(np.uint8)  # Convert to uint8

    # Use OpenCV morphological operations for fast min/max filtering
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    local_min = cv2.erode(img_gray, kernel)
    local_max = cv2.dilate(img_gray, kernel)

    # Compute contrast map
    contrast = (local_max - local_min).astype(np.float32) / 255.0

    # Normalize contrast to [0, 1]
    contrast -= contrast.min()
    contrast /= contrast.max() + 1e-8  # Prevent division by zero

    # Thresholding
    contrast[contrast > 0.75] = 0  # Remove extreme high contrast regions

    # contrast[contrast < 0.2] = 0  # Optional: Remove low contrast regions
    # contrast[contrast > 0] = 1  # Optional: Binarization
    contrast = np.expand_dims(contrast, axis=-1)

    return contrast

def genSobel(image, edge_threshold = 0.5, ksize = 5):
    # Compute edge detection using Sobel filter
    img_gray = (image).astype(np.uint8)  # Convert back to uint8
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=ksize)
    edges = np.sqrt(sobelx**2 + sobely**2)
    edges = (edges - edges.min()) / (edges.max() - edges.min())  # Normalize

    # Apply threshold
    edges[edges < edge_threshold] = 0  # Keep only strong edges

    edges = np.expand_dims(edges, axis=-1)
        
    return edges

# --- Predict and Stitch Function ---
def predict_and_stitch(image, model, window_size=(128, 128), stride=64, imageMasking = False, visualize = False, app = None, rgb = False):
    """
    Predicts on sliding window patches and stitches results together.
    
    Args:
        image: np.ndarray, input image.
        model: Trained TensorFlow/Keras model for prediction.
        window_size: tuple, size of each window (height, width).
        stride: int, step size for sliding window.
        
    Returns:
        Stitched output (e.g., segmentation mask) of the full image.
    """
    
    
    croph, cropw = image.shape[:2]

    

    if imageMasking:

        if image.ndim == 2:  # Check if image is 2D (H, W)
            image = image[:, :, np.newaxis]
        
        #imageMask = np.ones(image.shape)
        #image = np.concatenate((image, imageMask), axis=2)
        edges = genContrastEdges2(image, kernel_size=5) * 3
        edges = np.clip(edges,0,1)

    # Get patches and coordinates
    patches, coordinates, padIm = sliding_window(image, window_size, stride)       

    if imageMasking:
        edge_patches, _, _ = sliding_window(edges, window_size=window_size, stride=stride)
        patches = np.concatenate((patches, edge_patches), axis=3)     

    if rgb:
        patches = np.stack([patches] * 3, axis=3)


    h, w = padIm.shape[:2]
    stitched_output = np.zeros((h, w))  # Output canvas (e.g., mask)
    weight_matrix = np.zeros((h, w))   # For overlapping areas
    print("Shape of input:", patches.shape)

    totalPatches = patches.shape[0]
    currPatch = 0
    if app:
        app.PIPELINESUBPROGRESS = 0
    # Process each patch
    for patch, (y, x) in zip(patches, coordinates):
        # Preprocess the patch (normalize and expand dimensions)

        if visualize and app:
            app.AiProcessRect = [y, x, y + window_size[0], x + window_size[1]]
            print(app.AiProcessRect)

            if "c" in app.keypress_held_down:
                print("Aborting...")
                app.notify("Aborting AI process...")
                app.AiProcessRect = None
                return None
            
        if app:
            app.PIPELINESUBPROGRESS = currPatch/totalPatches

        patch_input = patch / 255.0  # Normalize
        if len(patch_input.shape) == 2:  # Add channel dimension for grayscale
            patch_input = np.expand_dims(patch_input, axis=-1)
        patch_input = np.expand_dims(patch_input, axis=0)  # Add batch dimension
        
        # Predict on the patch
        prediction = model.predict(patch_input, verbose=0)[0, :, :, 0]  # Assume single-channel output
        
        # Add prediction to the stitched output
        stitched_output[y:y+window_size[0], x:x+window_size[1]] += prediction
        weight_matrix[y:y+window_size[0], x:x+window_size[1]] += 1
        currPatch += 1

    # Normalize overlapping regions
    stitched_output /= np.maximum(weight_matrix, 1)  # Avoid division by zero
    if app:
        app.AiProcessRect = None

    return stitched_output[0:croph, 0:cropw]


# --- Main Execution ---
def main():
    # Load your trained TensorFlow model
    model_path = "C:/Users/cgvisa/Documents/VSCode/NEURAL NETWORKS/ITER20_MODELS.unet_upscale_860IMAGES_SHAPE_256x256x1_1.keras"  # Path to the trained model
    model = load_model(model_path)
    input_shape = model.input_shape[1:3]   # (H, W)
    output_shape = model.output_shape[1:3] # (H', W')
    print(input_shape, output_shape)
    input()
    
    # Load the large input image
    image_path = "kaikki2/14_3_23.tif"  # Path to the input image
    image = tf.keras.preprocessing.image.load_img(image_path, color_mode="grayscale")
    image = np.array(image)  # Convert to numpy array

    print(f"Input Image Shape: {image.shape}")
    
    # Process the image using sliding window
    print("Processing image using sliding window...")
    result = predict_and_stitch(image, model, window_size=(128, 128), stride=64)
    
    # Visualize the results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap="gray")

    plt.subplot(1, 2, 2)
    plt.title("Predicted Output")
    plt.imshow(result, cmap="gray")
    plt.colorbar()
    plt.show()

    # Optionally, save the stitched output
    output_path = "stitched_output.png"
    plt.imsave(output_path, result, cmap="gray")
    print(f"Stitched output saved to: {output_path}")

if __name__ == "__main__":
    main()
