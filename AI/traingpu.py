import tensorflow as tf
from tensorflow.keras import mixed_precision
import sys




# GPU Configuration
def setup_gpu():
    # Memory growth
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"Found {len(physical_devices)} GPU(s)")
        
        # Enable mixed precision
        mixed_precision.set_global_policy('mixed_float16')
        print("Mixed precision enabled")
        return True
    else:
        print("No GPU found!")
        return False

def main():
    # Setup GPU
    gpu_available = setup_gpu()
    sys.exit()
    # Your data loading code here...
    images, masks = load_data(image_dir, mask_dir, img_size=(128, 128))
    
    # Create efficient tf.data pipeline
    BATCH_SIZE = 32 if gpu_available else 16
    AUTOTUNE = tf.data.AUTOTUNE
    
    # Convert to tensorflow dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = (train_dataset
        .cache()
        .shuffle(1000)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE))
    
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    
    # Create and compile model
    model = build_enhanced_unet(input_shape=(128, 128, 1))
    
    # Use mixed precision optimizer if GPU available
    if gpu_available:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
            tf.keras.optimizers.Adam(learning_rate=1e-4)
        )
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', dice_coefficient]
    )
    
    # Train
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=50,
        callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(patience=5),
            tf.keras.callbacks.EarlyStopping(patience=10)
        ]
    )


if __name__ == "__main__":
    main()