from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
from ai_core import predict_and_stitch
import matplotlib.pyplot as plt
from MODELS.unet2 import dice_coefficient, dice_loss
import scipy.ndimage as sc
from skimage.measure import label, regionprops 
import pandas as pd
import cv2


def export(binary_image, csvName = ""):
    # Ensure the image is binary (convert to 0s and 1s)

    binary_image = (binary_image / np.max(binary_image)).astype(np.uint8)

    #_, binary_image = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)
    #binary_image = binary_image // 255  # Convert 255 to 1 for easier processing

    # Step 2: Label connected components
    labeled_image = label(binary_image)

    # Step 3: Measure particle properties
    properties = regionprops(labeled_image)


    

    # Step 4: Extract desired particle data
    particle_data = []
    for prop in properties:
        area = prop.area
        perimeter = prop.perimeter if prop.perimeter > 0 else 1  # Avoid division by zero
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        centroid = prop.centroid

        # Append data as a dictionary
        particle_data.append({
            "Area": area,
            "Perimeter": perimeter,
            "Circularity": circularity,
            "Centroid_X": centroid[0],
            "Centroid_Y": centroid[1]
        })

    # Step 5: Create a DataFrame
    df = pd.DataFrame(particle_data)
    
    # Step 6: Export to CSV
    output_csv = csvName
    df.to_csv(output_csv, index=False, sep=';', decimal=',')
    print(df)
    
    print(f"Particle data exported to {output_csv}")

# Helper functions for the loss and metrics
def dice_coefficient(y_true, y_pred):
    smooth = 1e-15
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)


def main():
    # Load your trained TensorFlow model
    model_path = "C:/Users/cgvisa/Documents/VSCode/NEURAL NETWORKS/ITER10_retrain_1619_images_1.keras"  # Path to the trained model
    model = load_model(model_path)
    model.summary()

    image_path = "C:/Users/cgvisa/Documents/VSCode/AI/BM_18_1_06.png"  # Path to the input image
    image_path2 = "C:/Users/cgvisa/Documents/VSCode/AI/BM_18_1_06D.png"  # Path to the input image

    image = tf.keras.preprocessing.image.load_img(image_path, color_mode="grayscale")
    image2 = tf.keras.preprocessing.image.load_img(image_path2, color_mode="grayscale")
    image = np.array(image)
    image2 = np.array(image2)
    imagePredicted = predict_and_stitch(image, model, window_size=(256, 256), stride=256, imageMasking=False)

    orig = (imagePredicted*255).astype(np.uint8)

    # Assuming image2 and imagePredicted are your input arrays
    # Threshold the images to binary (values 0 or 1)
    threshold_value = 127

    binary_image2 = (image2 > threshold_value).astype(np.uint8)  # Pixels > 127 become 1, else 0
    binary_imagePredicted = (imagePredicted*255 > threshold_value).astype(np.uint8)

    # Step 2: Label connected components

    particleLists = [{}, {}]
    for i, x in enumerate([binary_image2, binary_imagePredicted]):
        export(x, csvName = ["bm1.csv", "bm2.csv"][i])
        labeled_image = label(x)

        # Step 3: Measure particle sizes
        properties = regionprops(labeled_image)

        size_ranges = [
            (0, 10, "0-10 pixels"),
            (10, 20, "10-20 pixels"),
            (20, 50, "20-50 pixels"),
            (50, np.inf, "50+ pixels"),
        ]

        # Step 4: Filter particles by size
        min_size = 1  # Minimum particle size in pixels
        max_size = 20  # Maximum particle size in pixels
        particle_count = 0

        for prop in properties:
            if min_size <= prop.area <= max_size:
                particle_count += 1

            for sRange in size_ranges:

                if sRange[2] not in particleLists[i]:
                        particleLists[i][sRange[2]] = 0

                if sRange[0] <= prop.area <= sRange[1]:
                    
                    particleLists[i][sRange[2]] += 1

                    break

        print("UNDER 5 pixel sized particles:", particle_count)
    
    print(particleLists)



    difference = 1 - (binary_image2 == binary_imagePredicted)

    # Calculate the total number of pixels
    total_pixels = binary_image2.size  # Or image1.shape[0] * image1.shape[1]

    # Count the number of matching pixels
    matching_pixels = np.sum(binary_image2 == binary_imagePredicted)

    # Calculate accuracy
    accuracy = (matching_pixels / total_pixels) * 100

    print(f"Accuracy: {accuracy:.2f}%")

    # Print to verify (optional)
    print(f"Binary image2:\n{binary_image2}")
    print(f"Binary imagePredicted:\n{binary_imagePredicted}")
    print(imagePredicted.shape, image2.shape, np.max(imagePredicted), np.max(image2))

    # Visualize the results
    plt.figure(figsize=(18, 6))  # Adjust the width to fit three images comfortably
    plt.subplot(2, 2, 1)  # 1 row, 3 columns, first subplot
    plt.title("Original Image")
    plt.imshow(image, cmap="gray")
    plt.axis('off')  # Optional: Turn off the axes for better visualization

    plt.subplot(2, 2, 2)  # 1 row, 3 columns, second subplot
    plt.title("Predicted Particles")
    plt.imshow(orig, cmap="gray")
    plt.axis('off')  # Optional

    plt.subplot(2, 2, 3)  # 1 row, 3 columns, third subplot
    plt.title("Real Particles")
    plt.imshow(binary_image2, cmap="gray")
    plt.axis('off')  # Optional

    plt.subplot(2, 2, 4)
    plt.title("Differences")
    plt.imshow(difference, cmap="gray")
    plt.axis('off')  # Optional
    

    plt.tight_layout()  # Adjusts spacing between subplots for better appearance
    plt.show()




if __name__ == "__main__":
    main()