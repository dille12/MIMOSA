import numpy
import os
from ai_core import sliding_window


def load_upscaling_data(image_dir, patch_size=256, scale_factor=0.5, stride=128, max_images=0, rgb_data=False):
    """
    Loads images, extracts patches, downscales them for input, and uses original patches as ground truth.
    
    Args:
        image_dir: Path to the images folder.
        patch_size: Integer, size of the square patches to extract.
        scale_factor: Float, downscaling factor (0.5 means half resolution).
        stride: Integer, stride for patch extraction.
        max_images: Integer, maximum number of images to process (0 for all).
        rgb_data: Boolean, whether to convert grayscale to RGB.
        
    Returns:
        inputs: np.ndarray, array of downscaled image patches.
        targets: np.ndarray, array of original image patches.
    """
    inputs = []
    targets = []
    
    image_files = sorted(os.listdir(image_dir))
    i = 0
    
    for img_file in image_files:
        # Load image
        img_path = os.path.join(image_dir, img_file)
        
        # Load image in grayscale and normalize
        img = load_img(img_path, color_mode="grayscale")
        img = img_to_array(img) / 255.0
        
        # Extract patches for ground truth
        patches, _, _ = sliding_window(img, window_size=(patch_size, patch_size), stride=stride)
        
        # Process each patch
        for patch in patches:
            # Original patch is the target (ground truth)
            target = patch.copy()
            
            # Create downscaled version (input)
            # Calculate downscaled dimensions
            low_res_size = (int(patch_size * scale_factor), int(patch_size * scale_factor))
            
            # Resize down
            low_res_patch = tf.image.resize(patch, low_res_size).numpy()
            
            # Resize back up to original size (this creates the low-quality upscaled input)
            input_patch = tf.image.resize(low_res_patch, (patch_size, patch_size)).numpy()
            
            # Convert to RGB if needed
            if rgb_data:
                input_patch = np.stack([input_patch] * 3, axis=-1)
                target = np.stack([target] * 3, axis=-1)
            
            inputs.append(input_patch)
            targets.append(target)
        
        i += 1
        if max_images > 0 and i >= max_images:
            break
    
    # Convert to numpy arrays
    inputs = np.array(inputs)
    targets = np.array(targets)
    
    # Ensure correct shape (add channel dimension if needed)
    if len(inputs.shape) == 3:
        inputs = np.expand_dims(inputs, axis=-1)
    if len(targets.shape) == 3:
        targets = np.expand_dims(targets, axis=-1)
    
    print(f"Total patches - Inputs: {inputs.shape}, Targets: {targets.shape}")
    return inputs, targets