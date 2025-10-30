
from tensorflow.keras import layers, Model
from tensorflow import keras

# --- U-Net Model ---
def build_unet(input_shape=(128, 128, 1)):
    inputs = layers.Input(input_shape)

    # Encoder
    # Block 1
    c1 = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(inputs)
    print(f"c1 shape: {c1.shape}")


    c1 = layers.Dropout(0.1)(c1)
    c1 = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    # Block 2
    c2 = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(p1)
    c2 = layers.Dropout(0.1)(c2)
    c2 = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    # Block 3
    c3 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(p2)
    c3 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # Bottleneck
    c4 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(p3)
    c4 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(c4)

    # Decoder
    # Block 1
    u1 = layers.UpSampling2D((2, 2))(c4)
    u1 = layers.concatenate([u1, c3])
    c5 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(u1)
    c5 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(c5)

    # Block 2
    u2 = layers.UpSampling2D((2, 2))(c5)
    u2 = layers.concatenate([u2, c2])
    c6 = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(u2)
    c6 = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(c6)

    # Block 3
    u3 = layers.UpSampling2D((2, 2))(c6)
    u3 = layers.concatenate([u3, c1])
    c7 = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(u3)
    c7 = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(c7)

    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(c7)

    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

if __name__ == "__main__":
    from tensorflow.keras.utils import plot_model
    # Build the U-Net model
    model = build_unet(input_shape=(256,256,1))

    model.summary()