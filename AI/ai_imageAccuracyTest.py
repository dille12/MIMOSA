import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from ai_core import sliding_window
from MODELS import unet1, unet2, unet3, unetTEST
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from train import load_data

from ai_core import predict_and_stitch

def main():

    totValues = []

    for i in range(10):

        # Paths to images and masks
        accuracies = []
        patches = []
        for x in range(10):

            INPUT_SHAPE = 256
            STRIDE = 128
            INPUT_DEPTH = 2

            image_dir = "AI/train_images/"  # Replace with your folder path
            mask_dir = "AI/validate_images/"
            
            images, masks = load_data(image_dir, mask_dir, img_size=(INPUT_SHAPE, INPUT_SHAPE, INPUT_DEPTH), stride = STRIDE, maxImages=3*(x+1))



            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

            # Prepare train & validation datasets

            
            # Create model
            model = unetTEST.build_unet(input_shape=(INPUT_SHAPE, INPUT_SHAPE, INPUT_DEPTH))
            
            # Callbacks
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                        patience=5, min_lr=0.00001)
            early_stop = EarlyStopping(monitor='val_loss', patience=10, 
                                        restore_best_weights=True)
            # Training

            if True:

                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    batch_size=16,
                    epochs=10,
                    callbacks=[reduce_lr, early_stop],
                )
                print(history.params)

                image_path = "C:/Users/cgvisa/Documents/VSCode/AI/BM_18_1_06.png"  # Path to the input image
                image_path2 = "C:/Users/cgvisa/Documents/VSCode/AI/BM_18_1_06D.png"  # Path to the input image

                image = tf.keras.preprocessing.image.load_img(image_path, color_mode="grayscale")
                image2 = tf.keras.preprocessing.image.load_img(image_path2, color_mode="grayscale")
                image = np.array(image)
                image2 = np.array(image2).astype(np.float32) / 255
                imagePredicted = predict_and_stitch(image, model, window_size=(INPUT_SHAPE, INPUT_SHAPE), stride=STRIDE, imageMasking=True).astype(np.float32)
                print(np.max(imagePredicted))
                print(np.max(image))
                print(np.max(image2))

                accuracy = 1 - (np.sum(np.abs(imagePredicted - image2)) / (image2.size * np.max(image2)))
                patches.append(len(images))
                accuracies.append(float(accuracy))
                print(patches)
                print(accuracies)
                

            #plt.imshow(imagePredicted, cmap="gray")
            #plt.show()
        
        totValues.append([patches, accuracies])
        print(totValues)




if __name__ == "__main__":
    main()


