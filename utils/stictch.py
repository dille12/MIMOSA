from PIL import Image

def stitch_images(image1_path, image2_path, output_path):
    # Open the images
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    # Find the height of the taller image
    height = max(image1.height, image2.height)

    # Resize images to have the same height
    image1 = image1.resize((int(image1.width * height / image1.height), height))
    image2 = image2.resize((int(image2.width * height / image2.height), height))

    # Calculate the total width of the combined image
    total_width = image1.width + image2.width

    # Create a new blank image with the appropriate size
    stitched_image = Image.new('RGB', (total_width, height))

    # Paste the two images side by side
    stitched_image.paste(image1, (0, 0))
    stitched_image.paste(image2, (image1.width, 0))

    # Save the stitched image
    stitched_image.save(output_path)
    print(f"Stitched image saved as {output_path}")



# Example usage
# Replace 'image1.jpg' and 'image2.jpg' with the paths to your images
stitch_images("C:/Users/cgvisa/Documents/VSCode/utils/p1.png", "C:/Users/cgvisa/Documents/VSCode/utils/p2.png", 'stictch2.jpg')