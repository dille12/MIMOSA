import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from ai_core import sliding_window, genSobel, genContrastEdges, genContrastEdges2
from MODELS import unet1, unet2, unet3

os.environ['SM_FRAMEWORK'] = 'tf.keras'

from segmentation_models import Unet
from segmentation_models import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score

from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# --- Custom Data Generator Class ---
class SegmentationDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_filenames, mask_filenames, batch_size=8, img_size=(256, 256), 
                 stride=128, preprocess_fn=None, augment=True, shuffle=True, rgb_data=False):
        self.image_filenames = image_filenames
        self.mask_filenames = mask_filenames
        self.batch_size = batch_size
        self.img_size = img_size
        self.stride = stride
        self.preprocess_fn = preprocess_fn
        self.augment = augment
        self.shuffle = shuffle
        self.rgb_data = rgb_data
        self.indexes = np.arange(len(self.image_filenames))
        self.on_epoch_end()
        
        # Pre-calculate the number of patches per image to allocate batch arrays correctly
        img_h, img_w = img_to_array(load_img(self.image_filenames[0], color_mode="grayscale")).shape[:2]
        patches_h = (img_h - img_size[0]) // stride + 1
        patches_w = (img_w - img_size[1]) // stride + 1
        self.patches_per_image = patches_h * patches_w
        
    def __len__(self):
        """Number of batches per epoch"""
        return int(np.ceil(len(self.image_filenames) * self.patches_per_image / self.batch_size))
    
    def __getitem__(self, index):
        """Generate one batch of data"""
        # Calculate which image(s) to process for this batch
        start_img_idx = (index * self.batch_size) // self.patches_per_image
        end_img_idx = min(((index + 1) * self.batch_size + self.patches_per_image - 1) // self.patches_per_image, len(self.image_filenames))
        
        # Initialize batch arrays
        input_depth = 3 if self.rgb_data else 1
        X = []
        y = []
        
        # Process selected images into patches
        for i in range(start_img_idx, end_img_idx):
            img_file = self.image_filenames[self.indexes[i]]
            mask_file = self.mask_filenames[self.indexes[i]]
            
            # Load and normalize images
            img = load_img(img_file, color_mode="grayscale")
            img = img_to_array(img) / 255.0
            
            mask = load_img(mask_file, color_mode="grayscale")
            mask = img_to_array(mask) / 255.0
            
            # Extract patches
            img_patches, _, _ = sliding_window(img, window_size=self.img_size[:2] + (1,), stride=self.stride)
            mask_patches, _, _ = sliding_window(mask, window_size=self.img_size[:2] + (1,), stride=self.stride)
            
            # Convert to RGB if needed
            if self.rgb_data:
                img_patches = np.stack([img_patches] * 3, axis=3)
            
            # Preprocess if function provided
            if self.preprocess_fn:
                img_patches = self.preprocess_fn(img_patches)
            
            X.extend(img_patches)
            y.extend(mask_patches)
            
            # If we have enough for a batch, break early
            if len(X) >= self.batch_size:
                break
        
        # Take exactly batch_size items
        X = X[:self.batch_size]
        y = y[:self.batch_size]
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Apply augmentations
        if self.augment:
            X_aug = []
            y_aug = []
            for i in range(len(X)):
                img, mask = self._augment_pair(X[i], y[i])
                X_aug.append(img)
                y_aug.append(mask)
            X = np.array(X_aug)
            y = np.array(y_aug)
        
        return X, y
    
    def _augment_pair(self, image, mask):
        """Apply the same augmentation to both image and mask"""
        # Convert to tensors
        image_tensor = tf.convert_to_tensor(image)
        mask_tensor = tf.convert_to_tensor(mask)
        
        # Random Flip
        if np.random.rand() > 0.5:
            image_tensor = tf.image.flip_left_right(image_tensor)
            mask_tensor = tf.image.flip_left_right(mask_tensor)

        if np.random.rand() > 0.5:
            image_tensor = tf.image.flip_up_down(image_tensor)
            mask_tensor = tf.image.flip_up_down(mask_tensor)

        # Random Rotation (90-degree increments)
        k = np.random.randint(0, 4)
        image_tensor = tf.image.rot90(image_tensor, k)
        mask_tensor = tf.image.rot90(mask_tensor, k)

        # Random Brightness (only for images)
        if np.random.rand() > 0.5:
            image_tensor = tf.image.random_brightness(image_tensor, max_delta=0.2)
        
        # Convert back to numpy
        return image_tensor.numpy(), mask_tensor.numpy()
    
    def on_epoch_end(self):
        """Shuffle indexes after each epoch if shuffle is set to True"""
        if self.shuffle:
            np.random.shuffle(self.indexes)

# --- Main Training Function ---
def main():
    # Configuration
    INPUT_SHAPE = 256
    STRIDE = 128
    INPUT_DEPTH = 1
    BATCH_SIZE = 8  # Smaller batch size to reduce memory usage
    RETRAINMODEL = ""  # Path to model for retraining
    MODELVERSION = unet1
    ITERATION = 13
    USE_SM = True
    RGBDATA = True

    BACKBONE = 'resnet34'
    preprocess_input = get_preprocessing(BACKBONE)

    # Paths to data
    image_dir = "AI/train_images/"
    mask_dir = "AI/validate_images/"
    
    # Get file paths instead of loading all images
    image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
    mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])
    
    # Split file paths into train and validation sets
    train_img_files, val_img_files, train_mask_files, val_mask_files = train_test_split(
        image_files, mask_files, test_size=0.2, random_state=42
    )
    
    # Create data generators
    train_generator = SegmentationDataGenerator(
        train_img_files, train_mask_files, 
        batch_size=BATCH_SIZE, 
        img_size=(INPUT_SHAPE, INPUT_SHAPE), 
        stride=STRIDE, 
        preprocess_fn=preprocess_input if USE_SM else None,
        rgb_data=RGBDATA
    )
    
    val_generator = SegmentationDataGenerator(
        val_img_files, val_mask_files, 
        batch_size=BATCH_SIZE, 
        img_size=(INPUT_SHAPE, INPUT_SHAPE), 
        stride=STRIDE, 
        preprocess_fn=preprocess_input if USE_SM else None,
        augment=False,  # No augmentation for validation
        rgb_data=RGBDATA
    )
    
    # Find the next available filename
    base_filename = f"ITER{ITERATION}"
    if RETRAINMODEL:
        base_filename += "_RETRAIN"
    else:
        base_filename += f"_{str(MODELVERSION.__name__)}"
    
    # Estimate total patches for filename
    test_img = load_img(image_files[0], color_mode="grayscale")
    img_array = img_to_array(test_img)
    h, w = img_array.shape[:2]
    patches_h = (h - INPUT_SHAPE) // STRIDE + 1
    patches_w = (w - INPUT_SHAPE) // STRIDE + 1
    total_patches = len(image_files) * patches_h * patches_w
    
    base_filename += f"_{total_patches}IMAGES"
    base_filename += f"_SHAPE_{INPUT_SHAPE}x{INPUT_SHAPE}x{INPUT_DEPTH}"
    
    extension = ".keras"
    counter = 1
    
    while os.path.exists(f"{base_filename}_{counter}{extension}"):
        counter += 1
    
    filename = f"{base_filename}_{counter}{extension}"
    print("FILE IS TO BE SAVED IN", filename)
    
    # Create model
    if USE_SM:
        print("Using pre-weighted model...")
        model = Unet(BACKBONE, encoder_weights='imagenet', input_shape=(INPUT_SHAPE, INPUT_SHAPE, 3 if RGBDATA else 1))
        model.compile('Adam', loss=bce_jaccard_loss, metrics=[iou_score])
    elif RETRAINMODEL:
        print("Retraining...")
        model = load_model(RETRAINMODEL)
        print("Model", RETRAINMODEL, "loaded.")
    else:
        input_shape = (INPUT_SHAPE, INPUT_SHAPE, 3 if RGBDATA else INPUT_DEPTH)
        model = MODELVERSION.build_unet(input_shape=input_shape)
    
    model.ITERATION = ITERATION
    print(f"Model iteration: {model.ITERATION}")
    model.summary()
    
    # Callbacks
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                 patience=5, min_lr=0.00001)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, 
                              restore_best_weights=True)
    
    # Training with generators
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=100,
        callbacks=[reduce_lr, early_stop],
    )
    
    # Save the model
    model.save(filename)
    print(f"Model saved as '{filename}'")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    # Check if using binary_accuracy or iou_score
    metric_name = "iou_score" if "iou_score" in history.history else "binary_accuracy"
    val_metric_name = f"val_{metric_name}"
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history[metric_name], label=f"Train {metric_name}")
    plt.plot(history.history[val_metric_name], label=f"Val {metric_name}")
    plt.title(f"Training and Validation {metric_name}")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{base_filename}_{counter}_history.png")
    plt.show()

if __name__ == "__main__":
    main()