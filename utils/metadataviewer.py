from PIL import Image
import tifffile

def read_metadata_pillow(file_path):
    """Reads EXIF metadata using Pillow."""
    with Image.open(file_path) as img:
        metadata = img.info
        exif_data = img.getexif()

        print("\n--- PIL Metadata ---")
        for key, value in metadata.items():
            print(f"{key}: {value}")

        if exif_data:
            print("\n--- EXIF Data ---")
            for tag_id, value in exif_data.items():
                tag = Image.ExifTags.TAGS.get(tag_id, tag_id)
                print(f"{tag}: {value}")
        else:
            print("No EXIF data found.")

def read_metadata_tifffile(file_path):
    """Reads TIFF tags using tifffile."""
    with tifffile.TiffFile(file_path) as tif:
        print("\n--- TiffFile Metadata ---")
        for page in tif.pages:
            tags = page.tags
            for tag in tags.values():
                name, value = tag.name, tag.value
                print(f"{name}: {value}")

if __name__ == "__main__":
    
    path_to_tiff = "S:/71201_ElCo/Personal datafolders/Viliam/Old SEM images/14_2/N_14_2_04.tif"  # Replace with your file path
    read_metadata_pillow(path_to_tiff)
    read_metadata_tifffile(path_to_tiff)
