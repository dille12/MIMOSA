import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import regionprops
import cv2
import matplotlib.pyplot as plt

def separate_particles(binary_image, min_circularity=0.7, min_distance=20):
    """
    Separates overlapping particles in a binary image using watershed segmentation
    and filters them based on circularity.
    
    Parameters:
    -----------
    binary_image : numpy.ndarray
        Binary input image where particles are marked as 1 and background as 0
    min_circularity : float
        Minimum circularity threshold (0 to 1) for particle detection
    min_distance : int
        Minimum distance between local maxima for particle separation
        
    Returns:
    --------
    labeled_particles : numpy.ndarray
        Image with separated particles labeled with different integers
    particle_props : list
        List of particle properties including area, perimeter, and circularity
    """
    # Ensure binary image is boolean
    binary_image = binary_image.astype(bool)
    
    # Calculate distance transform
    distance = ndimage.distance_transform_edt(binary_image)
    
    # Find local maxima
    coords = peak_local_max(distance, min_distance=min_distance, labels=binary_image)
    
    # Create markers for watershed
    markers = np.zeros_like(binary_image, dtype=int)
    for i, (x, y) in enumerate(coords, start=1):
        markers[x, y] = i
    
    # Perform watershed segmentation
    labeled_particles = watershed(-distance, markers, mask=binary_image)
    
    # Calculate properties for each particle
    particle_props = []
    for region in regionprops(labeled_particles):
        # Get particle contour
        particle_mask = (labeled_particles == region.label).astype(np.uint8)
        contours, _ = cv2.findContours(particle_mask, cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_NONE)
        
        # Calculate circularity
        area = region.area
        perimeter = cv2.arcLength(contours[0], True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        particle_props.append({
            'label': region.label,
            'area': area,
            'perimeter': perimeter,
            'circularity': circularity,
            'centroid': region.centroid
        })
    
    # Filter particles based on circularity
    valid_labels = [prop['label'] for prop in particle_props 
                   if prop['circularity'] >= min_circularity]
    
    # Create final output with only valid particles
    final_labels = np.zeros_like(labeled_particles)
    for i, label in enumerate(valid_labels, start=1):
        final_labels[labeled_particles == label] = i
        
    return final_labels, particle_props

def visualize_results(original_image, labeled_particles, particle_props):
    """
    Creates a visualization of the separated particles with different colors.
    
    Parameters:
    -----------
    original_image : numpy.ndarray
        Original binary image
    labeled_particles : numpy.ndarray
        Image with separated particles labeled with different integers
    particle_props : list
        List of particle properties
    
    Returns:
    --------
    visualization : numpy.ndarray
        RGB image showing separated particles in different colors
    """
    # Create random colors for visualization
    n_particles = len(np.unique(labeled_particles)) - 1  # Subtract 1 for background
    colors = np.random.rand(n_particles + 1, 3)
    colors[0] = [0, 0, 0]  # Background color
    
    # Create RGB visualization
    visualization = np.zeros((*original_image.shape, 3))
    for label in range(1, n_particles + 1):
        visualization[labeled_particles == label] = colors[label]
    
    return visualization

def plot_comparison(binary_image, labeled_particles, particle_props):
    """
    Plots the original binary image and the separated particles side by side.
    
    Parameters:
    -----------
    binary_image : numpy.ndarray
        Original binary image
    labeled_particles : numpy.ndarray
        Image with separated particles labeled with different integers
    particle_props : list
        List of particle properties
    """
    # Create the visualization of separated particles
    visualization = visualize_results(binary_image, labeled_particles, particle_props)
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot original binary image
    ax1.imshow(binary_image, cmap='gray')
    ax1.set_title('Original Binary Image')
    ax1.axis('off')
    
    # Plot separated particles
    ax2.imshow(visualization)
    ax2.set_title(f'Separated Particles (n={len(particle_props)})')
    ax2.axis('off')
    
    # Add particle labels and circularity values
    for prop in particle_props:
        y, x = prop['centroid']
        label = prop['label']
        circularity = prop['circularity']
        ax2.text(x, y, f'{label}\n{circularity:.2f}', 
                color='white', ha='center', va='center',
                bbox=dict(facecolor='black', alpha=0.7, pad=1))
    
    plt.tight_layout()
    plt.show()
    
    # Print particle statistics
    print("\nParticle Statistics:")
    print("-" * 50)
    for prop in particle_props:
        print(f"Particle {prop['label']}:")
        print(f"  Circularity: {prop['circularity']:.3f}")
        print(f"  Area: {prop['area']} pixels")
        print(f"  Perimeter: {prop['perimeter']:.1f} pixels")
        print()

