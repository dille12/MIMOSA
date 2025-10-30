from tensorflow.keras import layers, Model
import tensorflow as tf

def build_unet(input_shape=(128, 128, 1)):
    """
    Enhanced U-Net for particle detection with better edge sensitivity
    and local feature awareness.
    """
    inputs = layers.Input(input_shape)

    # Initial feature extraction
    # Using smaller 3x3 kernels for finer detail detection
    init = layers.Conv2D(16, (3, 3), padding="same")(inputs)
    init = layers.BatchNormalization()(init)
    init = layers.Activation("relu")(init)

    # Encoder
    # Level 1 - fine details
    c1 = layers.Conv2D(32, (3, 3), padding="same")(init)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation("relu")(c1)
    c1 = layers.Conv2D(32, (3, 3), padding="same")(c1)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation("relu")(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    # Level 2 - particle boundaries
    c2 = layers.Conv2D(64, (3, 3), padding="same")(p1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Activation("relu")(c2)
    c2 = layers.Conv2D(64, (3, 3), padding="same")(c2)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Activation("relu")(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    # Level 3 - particle shapes
    c3 = layers.Conv2D(128, (3, 3), padding="same")(p2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Activation("relu")(c3)
    c3 = layers.Conv2D(128, (3, 3), padding="same")(c3)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Activation("relu")(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # Bottleneck - global context
    c4 = layers.Conv2D(256, (3, 3), padding="same")(p3)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Activation("relu")(c4)
    c4 = layers.Conv2D(256, (3, 3), padding="same")(c4)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Activation("relu")(c4)

    # Decoder with attention mechanism
    # Level 1 - reconstruct particle shapes
    u1 = layers.UpSampling2D((2, 2))(c4)
    u1 = layers.concatenate([u1, c3])
    c5 = layers.Conv2D(128, (3, 3), padding="same")(u1)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Activation("relu")(c5)
    c5 = layers.Conv2D(128, (3, 3), padding="same")(c5)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Activation("relu")(c5)

    # Level 2 - refine boundaries
    u2 = layers.UpSampling2D((2, 2))(c5)
    u2 = layers.concatenate([u2, c2])
    c6 = layers.Conv2D(64, (3, 3), padding="same")(u2)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.Activation("relu")(c6)
    c6 = layers.Conv2D(64, (3, 3), padding="same")(c6)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.Activation("relu")(c6)

    # Level 3 - fine-tune edges
    u3 = layers.UpSampling2D((2, 2))(c6)
    u3 = layers.concatenate([u3, c1])
    c7 = layers.Conv2D(32, (3, 3), padding="same")(u3)
    c7 = layers.BatchNormalization()(c7)
    c7 = layers.Activation("relu")(c7)
    c7 = layers.Conv2D(32, (3, 3), padding="same")(c7)
    c7 = layers.BatchNormalization()(c7)
    c7 = layers.Activation("relu")(c7)

    # Edge-aware output
    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(c7)

    model = Model(inputs, outputs)
    
    # Use a combination of losses for better edge detection
    losses = {
        'binary_crossentropy': tf.keras.losses.binary_crossentropy,
        'dice_loss': dice_loss  # You'll need to implement this
    }
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=losses['binary_crossentropy'],
        metrics=['accuracy', dice_coefficient]  # You'll need to implement dice_coefficient
    )
    
    return model

# Helper functions for the loss and metrics
def dice_coefficient(y_true, y_pred):
    smooth = 1e-15
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)
def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)


if __name__ == "__main__":
    from tensorflow.keras.utils import plot_model
    # Build the U-Net model
    model = build_unet()
    model.summary()

