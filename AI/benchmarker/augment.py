import os
import sys

from PIL import Image
import numpy as np





ims = os.listdir("AI/benchmarker/test")
print(ims)
i = input("Name of image >")
if i not in ims:
    print(i)
    sys.exit()


FILENAME = i.split(".")[0]

validationImage = f"AI/benchmarker/validate/{FILENAME}.png"
valIm = Image.open(validationImage)  # Change to your image file

# Load the image
image = Image.open(f"AI/benchmarker/test/{i}")  # Change to your image file
image_array = np.array(image).astype(np.int16)

i = input("Colorgrade/noise")
if i == "colorgrade":
    amount = int(input("Amount"))

    image_array = np.clip(image_array + amount, 0, 255)
    i = input("Output name: ")
    image = Image.fromarray(image_array.astype(np.uint8))
    image.save(f"AI/benchmarker/test/{i}.tif")  # Change output filename as needed
    valIm.save(f"AI/benchmarker/validate/{i}.png")

elif i == "noise":
    amount = int(input("Amount"))
    noise = np.random.randint(-amount, amount, image_array.shape, dtype=np.int8)

    image_array = np.clip(image_array + noise, 0, 255)
    i = input("Output name: ")
    image = Image.fromarray(image_array.astype(np.uint8))

    image.save(f"AI/benchmarker/test/{i}.tif")  # Change output filename as needed
    valIm.save(f"AI/benchmarker/validate/{i}.png")