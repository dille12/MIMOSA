import numpy as np
import os
os.environ['SM_FRAMEWORK'] = 'tf.keras'
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow import keras
from tensorflow.keras.optimizers import Adam


from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from ai_core import sliding_window, genSobel, genContrastEdges, genContrastEdges2
from MODELS import unet1, unet2, unet3, unet_upscale, unet_segment, unet4
from loadCustomModel import load_custom_segmentation_model
import cv2
import benchmarker.runtests
import shutil

def zip_folder(folder_path, output_filename):
    shutil.make_archive(output_filename, 'zip', folder_path)
from segmentation_models import Unet
from segmentation_models import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
import segmentation_models as sm
from realTimeGraph import FileLoggerCallback, DiscordProgressCB
from tensorflow.keras.models import load_model
# --- Data Loader ---
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from skimage.transform import resize

def load_data(image_dir, mask_dir, img_size=(128, 128), stride=64, maxImages=0, RGBDATA=False, AUGMENT=False, CONTEXT_SCALE = 1):
    """
    Loads images and masks, extracts sliding window patches, and prepares both
    original and downscaled inputs for a dual-branch CNN/UNet.

    Args:
        image_dir: Path to the images folder.
        mask_dir: Path to the masks folder.
        img_size: Tuple, target patch size (height, width).
        stride: Sliding window stride.
        maxImages: Maximum number of images to load (0 = all).
        RGBDATA: If True, replicate grayscale channel three times.
        AUGMENT: If True, applies data augmentation.
        downscale_factor: Fractional scaling for the downscaled context image.

    Returns:
        images: np.ndarray, array of normal-resolution patches.
        downscaled_images: np.ndarray, array of downscaled (context) patches.
        masks: np.ndarray, array of mask patches.
    """
    
    images_local = []
    images_context = []
    masks = []

    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))

    for i, (img_file, mask_file) in enumerate(zip(image_files, mask_files)):
        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_file)

        img = img_to_array(load_img(img_path, color_mode="grayscale")) / 255.0
        #context = resize(img, (img_size[0]*CONTEXT_SCALE, img_size[1]*CONTEXT_SCALE))
        mask = img_to_array(load_img(mask_path, color_mode="grayscale")) / 255.0

        if AUGMENT:
            img, mask = augment(img, mask)

        mask_patches, _, _ = sliding_window(mask, window_size=img_size, stride=stride)
        img_patches, coordinates, _ = sliding_window(img, window_size=img_size, stride=stride)

        # Generate context patches
        context_patches = []
        h, w, _ = img.shape
        patch_h, patch_w = img_size
        half_patch_h = patch_h // 2
        half_patch_w = patch_w // 2
        ctx_half_h = half_patch_h * CONTEXT_SCALE
        ctx_half_w = half_patch_w * CONTEXT_SCALE

        for cy, cx in coordinates:
            cx = int(cx + patch_w/2)
            cy = int(cy + patch_h/2)

            # Coordinates of context crop
            x0 = cx - ctx_half_w
            y0 = cy - ctx_half_h
            x1 = cx + ctx_half_w
            y1 = cy + ctx_half_h

            # Determine padding if out of bounds
            pad_top = max(0, -y0)
            pad_left = max(0, -x0)
            pad_bottom = max(0, y1 - h)
            pad_right = max(0, x1 - w)

            # Crop and pad
            y0_clamped = max(0, y0)
            y1_clamped = min(h, y1)
            x0_clamped = max(0, x0)
            x1_clamped = min(w, x1)
            print(cx,cy)

            context_crop = img[y0_clamped:y1_clamped, x0_clamped:x1_clamped]

            if pad_top or pad_bottom or pad_left or pad_right:
                context_crop = np.pad(
                    context_crop,
                    ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                    mode='constant', constant_values=0
                )

            # Resize to local window size
            context_resized = resize(context_crop, img_size, anti_aliasing=True)
            context_patches.append(context_resized)

        context_patches = np.array(context_patches)[..., np.newaxis]

        # RGB stacking if required
        if RGBDATA:
            img_patches = np.repeat(img_patches, 3, axis=-1)
            context_patches = np.repeat(context_patches, 3, axis=-1)

        images_local.extend(img_patches)
        images_context.extend(context_patches)
        masks.extend(mask_patches)

        if maxImages and i + 1 >= maxImages:
            break

    images_local = np.array(images_local)
    images_context = np.squeeze(np.array(images_context), axis=-1)
    masks = np.squeeze(np.array(masks), axis=-1)

    print(f"Local: {images_local.shape}, Context: {images_context.shape}, Masks: {masks.shape}")
    
    #for x, y in zip(images_local, images_context):
    #    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    #    axs[0].imshow(x, cmap='gray')
    #    axs[0].set_title("Original patch")
    #    axs[0].axis('off')
    #    axs[1].imshow(y.squeeze(), cmap='gray')
    #    axs[1].set_title("Downscaled patch")
    #    axs[1].axis('off')
    #    plt.tight_layout()
    #    plt.show()

    #images = np.concatenate((images_local, images_context), 3)
    #print(f"Final: {images.shape}")

    return images_local, images_context, masks


def weighted_bce_loss(pos_weight=10.0):
    def loss(y_true, y_pred):
        epsilon = 1e-7
        y_pred = keras.backend.clip(y_pred, epsilon, 1 - epsilon)
        return -keras.backend.mean(pos_weight * y_true * keras.backend.log(y_pred) + (1 - y_true) * keras.backend.log(1 - y_pred))
    return loss

def load_segmentation_data(image_dir, INPUT_SHAPE):
    ims = os.listdir(image_dir)
    imsTrain = [im for im in ims if "VAL" not in im]

    inputs = []
    targets = []

    for x in imsTrain:
        trainIm = image_dir + "/" + x
        valIm = image_dir + "/" + "VAL_" + x

        img = load_img(valIm, color_mode="grayscale")
        img = img_to_array(img) / 255.0
        img = cv2.resize(img, (INPUT_SHAPE, INPUT_SHAPE))
        inputs.append(img)

        mask = load_img(trainIm, color_mode="grayscale")
        mask = img_to_array(mask) / 255.0
        mask = cv2.resize(mask, (INPUT_SHAPE, INPUT_SHAPE))
        targets.append(mask)

     # Convert to numpy arrays
    inputs = np.array(inputs)
    targets = np.array(targets)

    print(f"(SEGM) Total patches - Inputs: {inputs.shape}, Targets: {targets.shape}")
    return inputs, targets
    

        




def load_upscaling_data(image_dir, patch_size=256, scale_factor=0.5, stride=128, max_images=0, rgb_data=False, AUGMENT = False):
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
            input_patch = tf.image.resize(patch, low_res_size).numpy()
                        
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

def visualize_training_data(inputs, targets, num_samples=5):
    """
    Visualizes pairs of input (low resolution upscaled) and target (high resolution) images.
    
    Args:
        inputs: np.ndarray, array of input images.
        targets: np.ndarray, array of target images.
        num_samples: Integer, number of samples to visualize (default: 5).
    """
    import matplotlib.pyplot as plt
    
    # Determine how many samples to show
    num_samples = min(num_samples, len(inputs))
    
    for i in range(num_samples):
        # Create a figure with two subplots side by side
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Get input and target images
        input_img = inputs[-i]
        target_img = targets[-i]
        
        # Remove any extra dimensions for display
        if input_img.shape[-1] == 1:
            input_img = input_img.squeeze()
        if target_img.shape[-1] == 1:
            target_img = target_img.squeeze()
        
        # Display input image
        axes[0].imshow(input_img, cmap='gray')
        axes[0].set_title(f"Input (Low Resolution Upscaled)")
        axes[0].axis('off')
        
        # Display target image
        axes[1].imshow(target_img, cmap='gray')
        axes[1].set_title(f"Target (High Resolution)")
        axes[1].axis('off')
        
        plt.suptitle(f"Sample {i+1}/{num_samples}")
        plt.tight_layout()
        plt.show()
        




from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import os

def augment(image, mask):
    """
    Apply augmentations to both image and mask with added noise generation.
    
    Args:
        image: Input image tensor
        mask: Corresponding segmentation mask tensor
    
    Returns:
        Augmented image and mask
    """
    # Random Flip
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)

    # Random Rotation
    k = tf.random.uniform(shape=(), minval=0, maxval=4, dtype=tf.int32)  # 0,1,2,3 (90-degree increments)
    image = tf.image.rot90(image, k)
    mask = tf.image.rot90(mask, k)

    # Random Brightness (only for images)
    image = tf.image.random_brightness(image, max_delta=0.2)
    
    # Noise Augmentations
    noise_type = tf.random.uniform(shape=(), minval=0, maxval=3, dtype=tf.int32)
    
    if noise_type == 0:
        # Gaussian Noise
        noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.05, dtype=image.dtype)
        image = tf.clip_by_value(image + noise, 0, 1)
    
    elif noise_type == 1:
        # Salt and Pepper Noise
        noise_mask = tf.random.uniform(shape=tf.shape(image), minval=0, maxval=1)
        salt_mask = tf.cast(noise_mask < 0.05, image.dtype)
        pepper_mask = tf.cast(noise_mask > 0.95, image.dtype)
        
        salt_image = tf.ones_like(image)
        pepper_image = tf.zeros_like(image)
        
        image = (1 - salt_mask) * image + salt_mask * salt_image
        image = (1 - pepper_mask) * image + pepper_mask * pepper_image
    
    elif noise_type == 2:
        # Poisson Noise (shot noise)
        noise = tf.random.poisson(shape=tf.shape(image), lam=50.0)
        noise = tf.cast(noise, image.dtype) / 50.0
        image = tf.clip_by_value(image + noise, 0, 1)
    
    return image, mask

AUTOTUNE = tf.data.AUTOTUNE

def prepare_dataset(images, masks, batch_size=16):
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    dataset = dataset.map(augment, num_parallel_calls=AUTOTUNE)
    dataset = dataset.shuffle(100).batch(batch_size).prefetch(AUTOTUNE)
    return dataset

import sys
# --- Main Training Function ---
def main():
    # Paths to images and masks

    path = os.path.dirname(os.path.abspath(__file__))
    path = path.replace("\\", "/")
    path = path.removesuffix("/AI")

    INPUT_SHAPE = 384
    STRIDE = 384
    INPUT_DEPTH = 1
    RETRAINMODEL = ""
    MODELVERSION = unet4
    ITERATION = 18
    COMMENT = ""
    
    USE_SM = False
    RGBDATA = False
    AUGMENT = True
    ADD_DILATED = True
    BACKBONE = 'resnet34'
    preprocess_input = get_preprocessing(BACKBONE)

    SUPER_RESOLUTION = False
    PARTICLE_SEPARATION = False

    ADDITIONALTESTDATA = False
    SKIPTRAIN = False
    SENDDATA = True

    if RETRAINMODEL:
        RETRAINMODEL = path + "/NEURAL NETWORKS/" + RETRAINMODEL
        


    image_dir = "AI/train_images/"  # Replace with your folder path
    mask_dir = "AI/validate_images/"
    print("Loading data...")
    if PARTICLE_SEPARATION:
        images, masks = load_segmentation_data("AI/particleValidation", INPUT_SHAPE)

    elif not SUPER_RESOLUTION:
        images, context, masks = load_data(image_dir, mask_dir, img_size=(INPUT_SHAPE, INPUT_SHAPE), stride = STRIDE, 
                                  RGBDATA = RGBDATA, AUGMENT = AUGMENT, maxImages=1,
                                  CONTEXT_SCALE=4)

    else:
        images, masks  = load_upscaling_data(image_dir, patch_size=INPUT_SHAPE*2, scale_factor=0.5, stride = STRIDE*2)


    print(np.max(images), np.max(masks), np.average(images))
    if ADDITIONALTESTDATA:
        valImages, valMasks = load_data("AI/TEST_train_images", "AI/TEST_validation_images", img_size=(INPUT_SHAPE, INPUT_SHAPE, INPUT_DEPTH), stride = STRIDE)
        print("Additional images appended. New patch length:", len(valImages), len(valMasks), ". Train length:", len(images), len(masks))


    
    # Find the next available filename
    base_filename = f"ITER{ITERATION}"
    if RETRAINMODEL:
        base_filename += "_RETRAIN"
    elif USE_SM:
        base_filename += "_SM"
    else:
        base_filename += f"_{str(MODELVERSION.__name__)}"
    if COMMENT:
        base_filename += f"_{COMMENT}"
    base_filename += f"_SHAPE_{INPUT_SHAPE}x{INPUT_SHAPE}x{INPUT_DEPTH}"


    extension = ".keras"
    counter = 1

    while os.path.exists( "NEURAL NETWORKS/" + f"{base_filename}_{counter}{extension}"):
        counter += 1

    filename = f"{base_filename}_{counter}{extension}"
    print("FILE IS TO BE SAVED IN", filename)

    
    # Split data
    X_train_local, X_val_local, X_train_context, X_val_context, y_train, y_val = train_test_split(
        images, context, masks, test_size=0.2, random_state=42
    )

    if ADDITIONALTESTDATA:
        X_val = np.concatenate((X_val, valImages), axis=0)
        y_val = np.concatenate((y_val, valMasks), axis=0)

    y_train = y_train[..., np.newaxis]
    y_val = y_val[..., np.newaxis]

    # Prepare train & validation datasets

    
    # Create model
    print("Creating the model...")
    if USE_SM:
        print("Print using preweighted model...")
        model = Unet(BACKBONE, encoder_weights='imagenet')
        model.compile(
            'Adam',
            loss=sm.losses.bce_jaccard_loss,
            metrics=[sm.metrics.iou_score, "accuracy"],
        )
        X_train = preprocess_input(X_train)
        X_val = preprocess_input(X_val)

    elif RETRAINMODEL:
        print("Retraining...")
        model = load_custom_segmentation_model(RETRAINMODEL)
        print("Model", RETRAINMODEL, "loaded.")

        new_lr = 1e-4
        optimizer = Adam(learning_rate=new_lr)

        # Recompile the model
        model.compile(optimizer=optimizer, loss=weighted_bce_loss(pos_weight=15.0), metrics=["accuracy"])



    else:
        model = MODELVERSION.build_unet(input_shape=(INPUT_SHAPE, INPUT_SHAPE, INPUT_DEPTH))



    model.ITERATION = ITERATION

    print(model.ITERATION)
    #model.summary()
    # CHANGE THE METRIC
    #model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3), loss = 'binary_crossentropy', metrics=["accuracy"])
    
    # Callbacks
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                   patience=5, min_lr=0.00001)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, 
                                restore_best_weights=True)
    
    plot_cb = FileLoggerCallback() 
    DC_cb = DiscordProgressCB()


    # Training

    if not SKIPTRAIN:

        history = model.fit(
            [X_train_local, X_train_context],
            y_train,
            validation_data=([X_val_local, X_val_context], y_val),
            batch_size=16,
            epochs=100
        )
    

    filenameSave = "NEURAL NETWORKS/" + filename

        # Save the model
    model.save(filenameSave)

    print(f"Model saved as '{filename}'")

    if SENDDATA:

        #output_dir = benchmarker.runtests.main(filename)
        #zip_folder(output_dir, "RESULTS")

        #sendtodc.sendToDC(filename, history)
        pass

    



def testData():
    image_dir = "AI/particleValidation/"  
    inputs, targets = load_segmentation_data(image_dir=image_dir, INPUT_SHAPE=128)
    visualize_training_data(inputs, targets, num_samples=100)


if __name__ == "__main__":
    #filename = "ITER16_RETRAIN_4166IMAGES_SHAPE_256x256x1_1.keras"
    #sendtodc.sendToDC(filename, "RESULTS.zip")
    #testData()
    main()
    #load_segmentation_data("C:/Users/cgvisa/Documents/VSCode/AI/particleValidation")
