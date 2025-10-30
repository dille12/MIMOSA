from tensorflow.keras import layers, Model
import tensorflow as tf

def build_unet(pretrained_weights=None, input_shape=(256,256,1)):
    # --- LOCAL INPUT BRANCH ---
    local_in = layers.Input(input_shape, name="local_input")
    conv1_l = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(local_in)
    conv1_l = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1_l)
    pool1_l = layers.MaxPooling2D(pool_size=(2, 2))(conv1_l)

    conv2_l = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1_l)
    conv2_l = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2_l)
    pool2_l = layers.MaxPooling2D(pool_size=(2, 2))(conv2_l)

    conv3_l = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2_l)
    conv3_l = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3_l)
    pool3_l = layers.MaxPooling2D(pool_size=(2, 2))(conv3_l)

    conv4_l = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3_l)
    conv4_l = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4_l)
    drop4_l = layers.Dropout(0.5)(conv4_l)
    pool4_l = layers.MaxPooling2D(pool_size=(2, 2))(drop4_l)

    # --- CONTEXT INPUT BRANCH ---
    context_in = layers.Input(input_shape, name="context_input")
    conv1_c = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(context_in)
    conv1_c = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1_c)
    pool1_c = layers.MaxPooling2D(pool_size=(2, 2))(conv1_c)

    conv2_c = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1_c)
    conv2_c = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2_c)
    pool2_c = layers.MaxPooling2D(pool_size=(2, 2))(conv2_c)

    conv3_c = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2_c)
    conv3_c = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3_c)
    pool3_c = layers.MaxPooling2D(pool_size=(2, 2))(conv3_c)

    conv4_c = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3_c)
    conv4_c = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4_c)
    drop4_c = layers.Dropout(0.5)(conv4_c)
    pool4_c = layers.MaxPooling2D(pool_size=(2, 2))(drop4_c)

    # --- BOTTLENECK FUSION ---
    merged = layers.concatenate([pool4_l, pool4_c], axis=3)
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merged)
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = layers.Dropout(0.5)(conv5)

    # --- DECODER ---
    up6 = layers.Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(layers.UpSampling2D(size=(2,2))(drop5))
    merge6 = layers.concatenate([drop4_l, up6], axis=3)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = layers.Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(layers.UpSampling2D(size=(2,2))(conv6))
    merge7 = layers.concatenate([conv3_l, up7], axis=3)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = layers.Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(layers.UpSampling2D(size=(2,2))(conv7))
    merge8 = layers.concatenate([conv2_l, up8], axis=3)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = layers.Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(layers.UpSampling2D(size=(2,2))(conv8))
    merge9 = layers.concatenate([conv1_l, up9], axis=3)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = layers.Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = layers.Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=[local_in, context_in], outputs=conv10, name="unet4")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5)]
    )

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model

if __name__ == "__main__":
    model = unet4()
    model.summary()
