from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
import pandas as pd
import os
import cv2
from datetime import datetime
import seaborn as sns
from ai_core import predict_and_stitch
import time
import segmentation_models as sm
from tensorflow import keras


from loadCustomModel import load_custom_segmentation_model
os.environ['SM_FRAMEWORK'] = 'tf.keras'




def analyze_image(image_path, model, ground_truth_path=None, min_particle_size=8, RGB = True):
    """
    Analyze an image with the model and compare to ground truth if available.
    
    Args:
        image_path: Path to the input image
        model: Loaded neural network model
        ground_truth_path: Path to the ground truth image (optional)
        min_particle_size: Minimum particle size to consider (smaller particles are treated as noise)
        
    Returns:
        Dictionary with analysis results
    """
    results = {
        'image_path': image_path,
        'ground_truth_path': ground_truth_path,
        'min_particle_size': min_particle_size
    }
    
    # Load input image
    try:
        image = tf.keras.preprocessing.image.load_img(image_path, color_mode="grayscale")
        image = np.array(image)
        results['image_shape'] = image.shape
    except Exception as e:
        results['error'] = f"Error loading input image: {e}"
        return results
    
    # Make prediction
    try:
        imagePredicted = predict_and_stitch(image, model, window_size=(256, 256), stride=256, imageMasking=False, rgb = RGB)
        results['prediction_shape'] = imagePredicted.shape
    except Exception as e:
        results['error'] = f"Error during prediction: {e}"
        return results
    
    # Convert prediction to binary
    threshold_value = 127
    binary_predicted = (imagePredicted*255 > threshold_value).astype(np.uint8)
    
    # Analyze predicted particles
    predicted_particles = analyze_particles(binary_predicted, min_particle_size)
    results['predicted_particles'] = predicted_particles
    
    # If ground truth is available, compare predictions
    if ground_truth_path:
        try:
            ground_truth = tf.keras.preprocessing.image.load_img(ground_truth_path, color_mode="grayscale")
            ground_truth = np.array(ground_truth)
            binary_ground_truth = (ground_truth > threshold_value).astype(np.uint8)
            
            # Analyze ground truth particles
            ground_truth_particles = analyze_particles(binary_ground_truth, min_particle_size)
            results['ground_truth_particles'] = ground_truth_particles
            
            # Create filtered binary images (removing small particles)
            filtered_binary_ground_truth = filter_small_particles(binary_ground_truth, min_size=min_particle_size)
            filtered_binary_predicted = filter_small_particles(binary_predicted, min_size=min_particle_size)
            
            # Calculate difference and accuracy using filtered images
            difference = 1 - (filtered_binary_ground_truth == filtered_binary_predicted)
            total_pixels = filtered_binary_ground_truth.size
            matching_pixels = np.sum(filtered_binary_ground_truth == filtered_binary_predicted)
            accuracy = (matching_pixels / total_pixels) * 100
            
            results['accuracy'] = accuracy
            results['difference_mask'] = difference
            results['filtered_ground_truth'] = filtered_binary_ground_truth
            results['filtered_prediction'] = filtered_binary_predicted
            
            # Calculate additional metrics
            true_positives = np.sum((filtered_binary_ground_truth == 1) & (filtered_binary_predicted == 1))
            false_positives = np.sum((filtered_binary_ground_truth == 0) & (filtered_binary_predicted == 1))
            true_negatives = np.sum((filtered_binary_ground_truth == 0) & (filtered_binary_predicted == 0))
            false_negatives = np.sum((filtered_binary_ground_truth == 1) & (filtered_binary_predicted == 0))
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results['precision'] = precision
            results['recall'] = recall
            results['f1_score'] = f1_score
            results['true_positives'] = true_positives
            results['false_positives'] = false_positives
            results['true_negatives'] = true_negatives
            results['false_negatives'] = false_negatives
            
            # Calculate size-based metrics
            size_comparison = compare_size_distributions(ground_truth_particles['size_distribution'], 
                                                         predicted_particles['size_distribution'])
            results['size_comparison'] = size_comparison
            
            # Count noise particles (particles smaller than min_particle_size)
            results['noise_particles_ground_truth'] = count_noise_particles(binary_ground_truth, min_particle_size)
            results['noise_particles_predicted'] = count_noise_particles(binary_predicted, min_particle_size)
            
        except Exception as e:
            results['ground_truth_error'] = f"Error processing ground truth: {e}"
    
    return results

def filter_small_particles(binary_image, min_size=8):
    """
    Remove particles smaller than min_size from a binary image
    
    Args:
        binary_image: Binary image with particles
        min_size: Minimum particle size to keep
        
    Returns:
        Binary image with small particles removed
    """
    # Label connected components
    labeled_image = label(binary_image)
    
    # Create output image
    filtered_image = np.zeros_like(binary_image)
    
    # Process each region
    for region in regionprops(labeled_image):
        if region.area >= min_size:
            # Keep particles that are large enough
            filtered_image[labeled_image == region.label] = 1
    
    return filtered_image

def count_noise_particles(binary_image, min_size=8):
    """
    Count particles smaller than min_size (noise particles)
    
    Args:
        binary_image: Binary image with particles
        min_size: Minimum particle size to keep (smaller are noise)
        
    Returns:
        Number of noise particles
    """
    # Label connected components
    labeled_image = label(binary_image)
    
    # Count noise particles
    noise_count = 0
    for region in regionprops(labeled_image):
        if region.area < min_size:
            noise_count += 1
    
    return noise_count

def analyze_particles(binary_image, min_particle_size=8):
    """
    Analyze particles in a binary image, skipping particles smaller than min_particle_size
    
    Args:
        binary_image: Binary image with particles
        min_particle_size: Minimum particle size to analyze (smaller particles are skipped as noise)
        
    Returns:
        Dictionary with particle analysis results
    """
    labeled_image = label(binary_image)
    properties = regionprops(labeled_image)
    
    # Filter out particles smaller than min_particle_size
    valid_particles = [prop for prop in properties if prop.area >= min_particle_size]
    noise_particles = [prop for prop in properties if prop.area < min_particle_size]
    
    result = {
        'total_particles': len(valid_particles),
        'noise_particles': len(noise_particles),
        'size_distribution': {},
        'particles_by_size': {},
        'average_size': 0 if len(valid_particles) == 0 else np.mean([prop.area for prop in valid_particles]),
        'median_size': 0 if len(valid_particles) == 0 else np.median([prop.area for prop in valid_particles]),
        'max_size': 0 if len(valid_particles) == 0 else max([prop.area for prop in valid_particles], default=0),
        'min_size': min_particle_size if len(valid_particles) > 0 else 0,
        'total_area': sum([prop.area for prop in valid_particles])
    }
    
    # Define size ranges
    size_ranges = [
        (min_particle_size, 20, f"{min_particle_size}-20 pixels"),
        (20, 50, "20-50 pixels"),
        (50, float('inf'), "50+ pixels"),
    ]
    
    # Count particles by size range
    for prop in valid_particles:
        for size_range in size_ranges:
            range_name = size_range[2]
            
            if range_name not in result['size_distribution']:
                result['size_distribution'][range_name] = 0
                result['particles_by_size'][range_name] = []
                
            if size_range[0] <= prop.area <= size_range[1]:
                result['size_distribution'][range_name] += 1
                result['particles_by_size'][range_name].append({
                    'area': prop.area,
                    'perimeter': prop.perimeter,
                    'eccentricity': prop.eccentricity,
                    'centroid': prop.centroid
                })
                break
    
    return result

def compare_size_distributions(ground_truth_dist, predicted_dist):
    """
    Compare size distributions between ground truth and predicted particles
    
    Args:
        ground_truth_dist: Ground truth size distribution
        predicted_dist: Predicted size distribution
        
    Returns:
        Dictionary with comparison metrics
    """
    result = {
        'absolute_difference': {},
        'percentage_difference': {},
        'total_difference': 0
    }
    
    # Ensure all categories exist in both distributions
    all_categories = set(list(ground_truth_dist.keys()) + list(predicted_dist.keys()))
    for category in all_categories:
        gt_count = ground_truth_dist.get(category, 0)
        pred_count = predicted_dist.get(category, 0)
        
        abs_diff = pred_count - gt_count
        result['absolute_difference'][category] = abs_diff
        
        if gt_count > 0:
            pct_diff = (abs_diff / gt_count) * 100
        elif pred_count > 0:
            pct_diff = float('inf')  # Ground truth has 0, but prediction has some
        else:
            pct_diff = 0  # Both are 0
            
        result['percentage_difference'][category] = pct_diff
        result['total_difference'] += abs(abs_diff)
    
    return result

def visualize_results(model, result, save_path=None, RGB = True):
    """
    Visualize analysis results for a single image
    
    Args:
        result: Dictionary with analysis results
        save_path: Path to save the visualization (optional)
    """
    if 'error' in result or 'ground_truth_error' in result:
        print(f"Error in results: {result.get('error', result.get('ground_truth_error'))}")
        return
    
    # Extract image paths
    image_path = result['image_path']
    ground_truth_path = result.get('ground_truth_path')
    min_particle_size = result.get('min_particle_size', 8)
    
    # Load images
    image = tf.keras.preprocessing.image.load_img(image_path, color_mode="grayscale")
    image = np.array(image)
    
    # Get predicted binary image
    threshold_value = 127
    imagePredicted = predict_and_stitch(image, model, window_size=(256, 256), stride=256, imageMasking=False, rgb=RGB)
    binary_predicted = (imagePredicted*255 > threshold_value).astype(np.uint8)
    filtered_predicted = filter_small_particles(binary_predicted, min_size=min_particle_size)
    
    # Setup visualization
    num_plots = 6 if ground_truth_path else 3
    fig, axes = plt.subplots(2, num_plots//2, figsize=(5*num_plots//2, 10))
    axes = axes.flatten()
    
    # Original image
    axes[0].set_title("Original Image")
    axes[0].imshow(image, cmap="gray")
    axes[0].axis('off')
    
    # Predicted image (raw)
    noise_count_pred = result['predicted_particles']['noise_particles']
    valid_count_pred = result['predicted_particles']['total_particles']
    axes[1].set_title(f"Raw Predicted\n{valid_count_pred + noise_count_pred} particles")
    axes[1].imshow(binary_predicted, cmap="gray")
    axes[1].axis('off')
    
    # Predicted image (filtered)
    axes[2].set_title(f"Filtered Predicted\n{valid_count_pred} particles\n{noise_count_pred} noise removed")
    axes[2].imshow(filtered_predicted, cmap="gray")
    axes[2].axis('off')
    
    if ground_truth_path:
        # Ground truth
        ground_truth = tf.keras.preprocessing.image.load_img(ground_truth_path, color_mode="grayscale")
        ground_truth = np.array(ground_truth)
        binary_ground_truth = (ground_truth > threshold_value).astype(np.uint8)
        filtered_ground_truth = filter_small_particles(binary_ground_truth, min_size=min_particle_size)
        
        noise_count_gt = result['ground_truth_particles']['noise_particles']
        valid_count_gt = result['ground_truth_particles']['total_particles']
        
        axes[3].set_title(f"Raw Ground Truth\n{valid_count_gt + noise_count_gt} particles")
        axes[3].imshow(binary_ground_truth, cmap="gray")
        axes[3].axis('off')
        
        axes[4].set_title(f"Filtered Ground Truth\n{valid_count_gt} particles\n{noise_count_gt} noise removed")
        axes[4].imshow(filtered_ground_truth, cmap="gray")
        axes[4].axis('off')
        
        # Difference
        difference = result['difference_mask']
        axes[5].set_title(f"Differences\nAccuracy: {result['accuracy']:.2f}%\nRecall: {result['recall']*100:.2f}%")
        axes[5].imshow(difference, cmap="hot")
        axes[5].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def create_summary_report(all_results, output_dir):
    """
    Create a summary report of all analyzed images
    
    Args:
        all_results: List of result dictionaries
        output_dir: Directory to save the report
    """
    # Create summary dataframe
    summary_data = []
    for result in all_results:
        if 'error' in result or 'ground_truth_error' in result:
            continue
            
        row = {
            'image_name': os.path.basename(result['image_path']),
            'accuracy': result.get('accuracy', None),
            'precision': result.get('precision', None),
            'recall': result.get('recall', None),
            'f1_score': result.get('f1_score', None),
            'gt_particles': result.get('ground_truth_particles', {}).get('total_particles', None),
            'pred_particles': result['predicted_particles']['total_particles'],
            'gt_noise_particles': result.get('noise_particles_ground_truth', None),
            'pred_noise_particles': result.get('noise_particles_predicted', None),
            'particle_diff': result.get('ground_truth_particles', {}).get('total_particles', 0) - 
                            result['predicted_particles']['total_particles'],
            'noise_diff': result.get('noise_particles_ground_truth', 0) - 
                         result.get('noise_particles_predicted', 0),
        }
        
        # Add size distribution stats
        if 'ground_truth_particles' in result:
            for size_range, count in result['ground_truth_particles']['size_distribution'].items():
                row[f'gt_{size_range}'] = count
                
        for size_range, count in result['predicted_particles']['size_distribution'].items():
            row[f'pred_{size_range}'] = count
            
        summary_data.append(row)
    
    # Create DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary to CSV
    summary_path = os.path.join(output_dir, 'summary_results.csv')
    summary_df.to_csv(summary_path, index=False)
    
    # Create summary visualizations if we have accuracy data
    if 'accuracy' in summary_df and not summary_df['accuracy'].isna().all():
        # Accuracy distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(summary_df['accuracy'], kde=True, bins=10)
        plt.title('Distribution of Accuracy Across Images')
        plt.xlabel('Accuracy (%)')
        plt.ylabel('Count')
        plt.savefig(os.path.join(output_dir, 'accuracy_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Precision-Recall plot
        plt.figure(figsize=(8, 8))
        plt.scatter(summary_df['precision'], summary_df['recall'], alpha=0.7)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel('Precision')
        plt.ylabel('Recall')
        plt.title('Precision vs Recall for All Images')
        plt.grid(True, linestyle='--', alpha=0.7)
        for i, row in summary_df.iterrows():
            plt.annotate(row['image_name'], 
                        (row['precision'], row['recall']),
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center')
        plt.savefig(os.path.join(output_dir, 'precision_recall.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    # Particle count comparison
    if 'gt_particles' in summary_df and not summary_df['gt_particles'].isna().all():
        plt.figure(figsize=(12, 6))
        summary_df = summary_df.sort_values('gt_particles')
        x = range(len(summary_df))
        plt.bar(x, summary_df['gt_particles'], width=0.4, label='Ground Truth', alpha=0.7)
        plt.bar([i+0.4 for i in x], summary_df['pred_particles'], width=0.4, label='Predicted', alpha=0.7)
        plt.xticks([i+0.2 for i in x], summary_df['image_name'], rotation=90)
        plt.xlabel('Image')
        plt.ylabel('Particle Count')
        plt.title('Ground Truth vs Predicted Particle Counts (Excluding Noise)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'particle_count_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Noise particles comparison
        if 'gt_noise_particles' in summary_df and not summary_df['gt_noise_particles'].isna().all():
            plt.figure(figsize=(12, 6))
            summary_df = summary_df.sort_values('gt_noise_particles')
            x = range(len(summary_df))
            plt.bar(x, summary_df['gt_noise_particles'], width=0.4, label='Ground Truth Noise', alpha=0.7)
            plt.bar([i+0.4 for i in x], summary_df['pred_noise_particles'], width=0.4, label='Predicted Noise', alpha=0.7)
            plt.xticks([i+0.2 for i in x], summary_df['image_name'], rotation=90)
            plt.xlabel('Image')
            plt.ylabel('Noise Particle Count')
            plt.title(f'Ground Truth vs Predicted Noise Particles (< 8 pixels)')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'noise_count_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
    # Size distribution analysis - skip the 0-8 size range that's now excluded
    min_particle_size = 8
    if all_results and all_results[0].get('min_particle_size'):
        min_particle_size = all_results[0].get('min_particle_size')
        
    # Dynamically determine size ranges from the results
    size_ranges = set()
    for result in all_results:
        if 'ground_truth_particles' in result:
            size_ranges.update(result['ground_truth_particles'].get('size_distribution', {}).keys())
        size_ranges.update(result['predicted_particles'].get('size_distribution', {}).keys())
    
    size_ranges = sorted(list(size_ranges))
    
    size_data = {
        'ground_truth': {size_range: [] for size_range in size_ranges},
        'predicted': {size_range: [] for size_range in size_ranges}
    }
    
    for _, row in summary_df.iterrows():
        for size_range in size_ranges:
            gt_key = f'gt_{size_range}'
            pred_key = f'pred_{size_range}'
            
            if gt_key in row and not pd.isna(row[gt_key]):
                size_data['ground_truth'][size_range].append(row[gt_key])
            else:
                size_data['ground_truth'][size_range].append(0)
                
            if pred_key in row:
                size_data['predicted'][size_range].append(row[pred_key])
            else:
                size_data['predicted'][size_range].append(0)
    
    # Create size distribution bar plot
    plt.figure(figsize=(10, 8))
    bar_width = 0.35
    index = np.arange(len(size_ranges))
    
    # Calculate averages
    gt_avgs = [np.mean(size_data['ground_truth'][size_range]) for size_range in size_ranges]
    pred_avgs = [np.mean(size_data['predicted'][size_range]) for size_range in size_ranges]
    
    # Plot
    plt.bar(index, gt_avgs, bar_width, label='Ground Truth', alpha=0.7)
    plt.bar(index + bar_width, pred_avgs, bar_width, label='Predicted', alpha=0.7)
    
    plt.xlabel('Particle Size Range')
    plt.ylabel('Average Count')
    plt.title(f'Average Particle Count by Size Range (Excluding < {min_particle_size} pixel noise)')
    plt.xticks(index + bar_width/2, size_ranges)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'size_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return summary_df


def test_model_robustness(model, image_path, ground_truth_path=None, output_dir=None, RGB = True):
    """
    Test the model's robustness by applying various image transformations
    
    Args:
        model: Trained segmentation model
        image_path: Path to the input image
        ground_truth_path: Path to the ground truth image (optional)
        output_dir: Directory to save results (optional)
        
    Returns:
        Dictionary with robustness test results
    """
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    print("Testing model robustness...")
    # Create output directory if not specified
    if output_dir is None:
        output_dir = os.path.dirname(image_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load original image
    original_image = tf.keras.preprocessing.image.load_img(image_path, color_mode="grayscale")
    original_image = np.array(original_image)
    
    # Load ground truth if available
    ground_truth = None
    if ground_truth_path:
        ground_truth = tf.keras.preprocessing.image.load_img(ground_truth_path, color_mode="grayscale")
        ground_truth = np.array(ground_truth)
    
    # Robustness tests to perform
    robustness_tests = [
        # Brightness tests
        {
            'name': 'Brightness',
            'variations': [
                {'factor': 0.5, 'label': 'Reduced Brightness'},
                {'factor': 1.5, 'label': 'Increased Brightness'},
                {'factor': 2.0, 'label': 'Highly Increased Brightness'}
            ],
            'transform': lambda img, factor: np.clip(img * factor, 0, 255).astype(np.uint8)
        },
        
        # Noise tests
        {
            'name': 'Gaussian Noise',
            'variations': [
                {'factor': 0.01, 'label': 'Low Noise'},
                {'factor': 0.05, 'label': 'Medium Noise'},
                {'factor': 0.1, 'label': 'High Noise'}
            ],
            'transform': lambda img, factor: np.clip(img + np.random.normal(0, factor * 255, img.shape), 0, 255).astype(np.uint8)
        },
        
        # Contrast tests
        {
            'name': 'Contrast',
            'variations': [
                {'factor': 0.5, 'label': 'Reduced Contrast'},
                {'factor': 1.5, 'label': 'Increased Contrast'},
                {'factor': 2.0, 'label': 'Highly Increased Contrast'}
            ],
            'transform': lambda img, factor: np.clip((img - np.mean(img)) * factor + np.mean(img), 0, 255).astype(np.uint8)
        }
    ]
    
    # Store results
    robustness_results = {
        'original_image_path': image_path,
        'ground_truth_path': ground_truth_path,
        'tests': {}
    }
    
    # Visualization setup
    num_tests = len(robustness_tests)
    num_variations = max(len(test['variations']) for test in robustness_tests)
    fig, axes = plt.subplots(num_tests, num_variations + 1, figsize=(5*(num_variations+1), 5*num_tests))
    
    # Perform robustness tests
    for test_idx, test in enumerate(robustness_tests):
        test_name = test['name']
        test_results = {}
        
        # Plot original image
        if num_tests > 1:
            current_axes = axes[test_idx]
        else:
            current_axes = axes
        
        current_axes[0].imshow(original_image, cmap='gray')
        current_axes[0].set_title(f'Original Image\n{test_name}')
        current_axes[0].axis('off')
        
        # Test each variation
        for var_idx, variation in enumerate(test['variations'], 1):
            # Apply transformation
            transformed_image = test['transform'](original_image, variation['factor'])
            
            # Predict on transformed image
            imagePredicted = predict_and_stitch(transformed_image, model, window_size=(256, 256), stride=256, imageMasking=False, rgb=RGB)
            
            # Binary prediction
            threshold_value = 127
            binary_predicted = (imagePredicted*255 > threshold_value).astype(np.uint8)
            
            # Analyze particles
            predicted_particles = analyze_particles(binary_predicted)
            
            # Compare with ground truth if available
            if ground_truth is not None:
                binary_ground_truth = (ground_truth > threshold_value).astype(np.uint8)
                filtered_ground_truth = filter_small_particles(binary_ground_truth)
                filtered_predicted = filter_small_particles(binary_predicted)


                
                
                # Calculate metrics
                true_positives = np.sum((filtered_ground_truth == 1) & (filtered_predicted == 1))
                false_positives = np.sum((filtered_ground_truth == 0) & (filtered_predicted == 1))
                true_negatives = np.sum((filtered_ground_truth == 0) & (filtered_predicted == 0))
                false_negatives = np.sum((filtered_ground_truth == 1) & (filtered_predicted == 0))
                
                accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                test_results[variation['label']] = {
                    'accuracy' : accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score
                }
            else:
                test_results[variation['label']] = {
                }
            
            # Visualize transformed image and prediction
            current_axes[var_idx].imshow(binary_predicted, cmap='gray')
            current_axes[var_idx].set_title(f"{variation['label']}\nParticles: {predicted_particles['total_particles']}")
            current_axes[var_idx].axis('off')
        
        robustness_results['tests'][test_name] = test_results
    
    # Save visualization
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'robustness_test_{os.path.splitext(os.path.basename(image_path))[0]}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Serialize and save detailed results
    import json
    results_path = os.path.join(output_dir, f'robustness_results_{os.path.splitext(os.path.basename(image_path))[0]}.json')
    with open(results_path, 'w') as f:
        json.dump(robustness_results, f, indent=2, default=str)
    
    return robustness_results


def main(MODEL, RGB):
    # Configuration
    model_path = f"C:/Users/cgvisa/Documents/VSCode/NEURAL NETWORKS/{MODEL}"
    print()
    test_dir = os.getcwd() + "/AI/benchmarker/test"
    validate_dir = os.getcwd() + "/AI/benchmarker/validate"
    
    # Create timestamp for output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"AI/benchmarker/results/{MODEL}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {model_path}...")

    model = load_custom_segmentation_model(model_path)

   # model = load_model(model_path, custom_objects={'binary_crossentropy_plus_jaccard_loss': sm.losses.bce_jaccard_loss})
    model.summary()
    
    # Get list of images
    test_images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.lower().endswith(('png', 'jpg', 'jpeg', 'tif', 'tiff'))]
    validate_images = [os.path.join(validate_dir, f) for f in os.listdir(validate_dir) if f.lower().endswith(('png', 'jpg', 'jpeg', 'tif', 'tiff'))]
    
    print(f"Found {len(test_images)} test images and {len(validate_images)} validation images")
    
    # Match test images with validation images (ground truth)
    image_pairs = []
    
    # Try to match by name (assuming similar naming convention)
    for test_img in test_images:
        test_basename = os.path.splitext(os.path.basename(test_img))[0]
        
        # Look for exact match first
        found_match = False
        for val_img in validate_images:
            val_basename = os.path.splitext(os.path.basename(val_img))[0]
            if test_basename == val_basename:
                image_pairs.append((test_img, val_img))
                found_match = True
                break
                
        # If no exact match, look for validation image with 'D' suffix (as in original code)
        if not found_match:
            for val_img in validate_images:
                val_basename = os.path.splitext(os.path.basename(val_img))[0]
                if test_basename + 'D' == val_basename:
                    image_pairs.append((test_img, val_img))
                    found_match = True
                    break
                    
        # If still no match, process test image without ground truth
        if not found_match:
            image_pairs.append((test_img, None))
    
    # Process all images
    all_results = []
    
    for i, (img_path, gt_path) in enumerate(image_pairs):
        print(f"Processing image {i+1}/{len(image_pairs)}: {os.path.basename(img_path)}")
        
        # Analyze image
        result = analyze_image(img_path, model, gt_path, RGB = RGB)
        all_results.append(result)
        
        # Visualize and save result
        vis_path = os.path.join(output_dir, f"result_{os.path.splitext(os.path.basename(img_path))[0]}.png")
        visualize_results(model, result, save_path=vis_path, RGB = RGB)
        
        # Save detailed results to JSON
        result_path = os.path.join(output_dir, f"details_{os.path.splitext(os.path.basename(img_path))[0]}.json")
        # Convert NumPy arrays and other non-serializable objects to lists or strings
        serializable_result = {}
        for key, value in result.items():
            if key == 'difference_mask' and value is not None:
                # Save difference mask as a separate image
                diff_path = os.path.join(output_dir, f"diff_{os.path.splitext(os.path.basename(img_path))[0]}.png")
                plt.imsave(diff_path, value, cmap='hot')
                serializable_result[key] = diff_path
            elif isinstance(value, np.ndarray):
                serializable_result[key] = value.shape
            elif isinstance(value, dict):
                serializable_result[key] = value
            else:
                serializable_result[key] = value
                
        # Save serializable result
        import json
        with open(result_path, 'w') as f:
            json.dump(serializable_result, f, indent=2, default=str)


        robustness_results = test_model_robustness(model, img_path, gt_path, output_dir, RGB = RGB)
        print(robustness_results)
    
    # Create summary report
    print("Creating summary report...")
    summary_df = create_summary_report(all_results, output_dir)
    
    # Print summary statistics
    if 'accuracy' in summary_df and not summary_df['accuracy'].isna().all():
        print("\n===== SUMMARY STATISTICS =====")
        print(f"Average Accuracy: {summary_df['accuracy'].mean():.2f}% ± {summary_df['accuracy'].std():.2f}%")
        print(f"Average Precision: {summary_df['precision'].mean():.4f}")
        print(f"Average Recall: {summary_df['recall'].mean():.4f}")
        print(f"Average F1 Score: {summary_df['f1_score'].mean():.4f}")
        
    if 'gt_particles' in summary_df and not summary_df['gt_particles'].isna().all():
        print("\n===== PARTICLE COUNTS =====")
        print(f"Average Ground Truth Particles: {summary_df['gt_particles'].mean():.1f}")
        print(f"Average Predicted Particles: {summary_df['pred_particles'].mean():.1f}")
        print(f"Average Absolute Difference: {summary_df['particle_diff'].abs().mean():.1f}")
        print(f"Average Relative Difference: {(summary_df['particle_diff'] / summary_df['gt_particles'] * 100).mean():.2f}%")
    
    print(f"\nAnalysis complete! Results saved to {output_dir}")
    return output_dir

if __name__ == "__main__":
    main()