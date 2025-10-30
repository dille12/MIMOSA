import cv2
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops


def export(binary_image):
    # Ensure the image is binary (convert to 0s and 1s)
    _, binary_image = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)
    binary_image = binary_image // 255  # Convert 255 to 1 for easier processing

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
    output_csv = 'particle_data.csv'
    df.to_csv(output_csv, index=False, decimal=',', )
    print(f"Particle data exported to {output_csv}")
