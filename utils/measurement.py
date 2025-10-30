import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import re
import os

def parse_metadata(metadata_file):
    """Parse the metadata file to extract field of view information."""
    with open(metadata_file, 'r', encoding="utf-8", errors="ignore") as f:
        content = f.read()
    
    # Extract field of view line - handle variable whitespace and format
    fov_match = re.search(r'\$CM_FIELD_OF_VIEW\s+([\d.]+)µm\s+([\d.]+)µm', content)
    if not fov_match:
        # Try alternative format with different spacing
        fov_match = re.search(r'\$CM_FIELD_OF_VIEW\s+([\d.]+)\s*µm\s+([\d.]+)\s*µm', content)
    
    if not fov_match:
        # Debug: print the actual line to see the format
        print("Available lines containing FIELD_OF_VIEW:")
        for line in content.split('\n'):
            if 'FIELD_OF_VIEW' in line:
                print(f"  '{line}'")
                print(f"  Bytes: {[ord(c) for c in line]}")
        raise ValueError("Could not find CM_FIELD_OF_VIEW in metadata")
    
    fov_width_um = float(fov_match.group(1))
    fov_height_um = float(fov_match.group(2))
    
    return fov_width_um, fov_height_um

def calculate_scale_bar_params(fov_width_um, image_width_px):
    """Calculate appropriate scale bar length and label."""
    # Calculate micrometers per pixel
    um_per_pixel = fov_width_um / image_width_px
    
    # Choose appropriate scale bar length (aim for 10-20% of image width)
    target_length_px = image_width_px * 0.15
    target_length_um = target_length_px * um_per_pixel
    
    # Round to nice numbers
    nice_values = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    scale_bar_um = min(nice_values, key=lambda x: abs(x - target_length_um))
    
    # Calculate actual pixel length
    scale_bar_px = int(scale_bar_um / um_per_pixel)
    
    return scale_bar_px, scale_bar_um

def add_scale_bar(image_path, metadata_path, output_path=None):
    """Add scale bar to SEM image based on metadata."""
    
    # Parse metadata
    fov_width_um, fov_height_um = parse_metadata(metadata_path)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    height, width = image.shape[:2]
    
    # Calculate scale bar parameters
    scale_bar_px, scale_bar_um = calculate_scale_bar_params(fov_width_um, width)
    
    # Scale bar styling
    bar_thickness = max(3, height // 200)  # Adaptive thickness
    margin = max(20, width // 50)  # Margin from edges
    font_size = max(12, height // 80)  # Adaptive font size
    
    # Position scale bar in bottom right
    bar_start_x = width - margin - scale_bar_px
    bar_end_x = width - margin
    bar_y = height - margin - bar_thickness - font_size - 10
    
    # Draw white scale bar with black outline
    cv2.rectangle(image, 
                  (bar_start_x - 1, bar_y - 1), 
                  (bar_end_x + 1, bar_y + bar_thickness + 1), 
                  (0, 0, 0), -1)  # Black outline
    
    cv2.rectangle(image, 
                  (bar_start_x, bar_y), 
                  (bar_end_x, bar_y + bar_thickness), 
                  (255, 255, 255), -1)  # White bar
    
    # Convert to PIL for text rendering
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    
    # Try to use a nice font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)  # macOS
        except:
            font = ImageFont.load_default()
    
    # Format scale bar label
    if scale_bar_um >= 1000:
        label = f"{scale_bar_um/1000:.0f} mm"
    elif scale_bar_um >= 1:
        label = f"{scale_bar_um:.0f} µm"
    else:
        label = f"{scale_bar_um*1000:.0f} nm"
    
    # Get text dimensions
    bbox = draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Center text above scale bar
    text_x = bar_start_x + (scale_bar_px - text_width) // 2
    text_y = bar_y - text_height - 5
    
    # Draw text with black outline for visibility
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx != 0 or dy != 0:
                draw.text((text_x + dx, text_y + dy), label, font=font, fill=(0, 0, 0))
    draw.text((text_x, text_y), label, font=font, fill=(255, 255, 255))
    
    # Convert back to opencv
    image_final = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    
    # Save image
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_with_scale{ext}"
    
    cv2.imwrite(output_path, image_final)
    
    print(f"Scale bar added successfully!")
    print(f"Field of view: {fov_width_um:.2f} x {fov_height_um:.2f} µm")
    print(f"Image size: {width} x {height} pixels")
    print(f"Scale: {fov_width_um/width:.3f} µm/pixel")
    print(f"Scale bar: {scale_bar_um} µm ({scale_bar_px} pixels)")
    print(f"Output saved to: {output_path}")
    
    return output_path


# Example usage
if __name__ == "__main__":
    # Replace with your file paths
    image_file = "C:/Users/cgvisa/Documents/VSCode/images divided by sample/VTT008/19-5_0024.tif"
    metadata_file = "C:/Users/cgvisa/Documents/VSCode/images divided by sample/VTT008/19-5_0024.txt"
    
    try:
        output_file = add_scale_bar(image_file, metadata_file)
    except Exception as e:
        print(f"Error: {e}")