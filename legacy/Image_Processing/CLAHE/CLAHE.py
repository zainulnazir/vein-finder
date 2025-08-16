import cv2
import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.gridspec import GridSpec

def apply_clahe(image_path, save_path, clip_limit=2.0, tile_grid_size=(8, 8), output_format='jpg', quality=95, create_visualizations=True):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) with comprehensive research visualizations.
    
    Args:
        image_path: Path to the input image
        save_path: Path to save the processed image or directory
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
        output_format: Output image format ('jpg', 'png', etc.)
        quality: JPEG quality (0-100), only applicable for JPEG
        create_visualizations: Whether to create research visualizations
    
    Returns:
        tuple: (clahe_image, metadata_dict)
    """
    # Check if input file exists
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Read the image in grayscale
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
    except Exception as e:
        raise RuntimeError(f"Error reading image: {str(e)}")

    # Apply CLAHE
    start_time = time.time()
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    clahe_image = clahe.apply(image)
    processing_time = time.time() - start_time

    # Ensure save_path has a valid filename and extension
    if os.path.isdir(save_path):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(save_path, f"{base_name}_clahe.{output_format}")
    else:
        # Ensure the directory exists
        output_dir = os.path.dirname(save_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

    # Save the processed image with appropriate parameters
    try:
        if output_format.lower() == 'jpg' or output_format.lower() == 'jpeg':
            cv2.imwrite(save_path, clahe_image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        elif output_format.lower() == 'png':
            cv2.imwrite(save_path, clahe_image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        else:
            cv2.imwrite(save_path, clahe_image)
    except Exception as e:
        raise RuntimeError(f"Error saving image: {str(e)}")

    # Initialize visualization paths
    comparison_path = None
    histogram_path = None
    intensity_path = None
    metrics_path = None

    # Create comparison image
    comparison_path = save_path.replace(f"_clahe.{output_format}", f"_comparison.{output_format}")
    create_comparison_image(image, clahe_image, comparison_path, output_format, quality)

    # Create research visualizations if requested
    if create_visualizations:
        # Create histogram comparison
        histogram_path = save_path.replace(f"_clahe.{output_format}", f"_histograms.png")
        create_histogram_plot(image, clahe_image, histogram_path)
        
        # Create intensity distribution plot
        intensity_path = save_path.replace(f"_clahe.{output_format}", f"_intensity_distribution.png")
        create_intensity_distribution(image, clahe_image, intensity_path)
        
        # Create metrics visualization
        metrics_path = save_path.replace(f"_clahe.{output_format}", f"_metrics.png")
        create_metrics_visualization(image, clahe_image, processing_time, clip_limit, tile_grid_size, metrics_path)

    # Get output file size
    output_stats = os.stat(save_path)
    
    # Calculate image statistics
    orig_mean = np.mean(image)
    orig_std = np.std(image)
    orig_min = np.min(image)
    orig_max = np.max(image)
    
    clahe_mean = np.mean(clahe_image)
    clahe_std = np.std(clahe_image)
    clahe_min = np.min(clahe_image)
    clahe_max = np.max(clahe_image)
    
    # Calculate contrast improvement
    contrast_improvement = ((clahe_std - orig_std) / orig_std) * 100 if orig_std > 0 else 0
    
    # Gather metadata
    input_stats = os.stat(image_path)
    height, width = clahe_image.shape
    info = {
        "Original Image Path": image_path,
        "Processed Image Path": save_path,
        "Comparison Image Path": comparison_path,
        "Histogram Plot Path": histogram_path,
        "Intensity Distribution Path": intensity_path,
        "Metrics Visualization Path": metrics_path,
        "Image Dimensions": (height, width),
        "Original File Size (KB)": round(input_stats.st_size / 1024, 2),
        "Processed File Size (KB)": round(output_stats.st_size / 1024, 2),
        "File Size Change (%)": round((output_stats.st_size / input_stats.st_size - 1) * 100, 2),
        "Image Depth": image.dtype,
        "Clip Limit": clip_limit,
        "Tile Grid Size": tile_grid_size,
        "Processing Time (ms)": round(processing_time * 1000, 2),
        "Processing Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Original Mean Intensity": round(orig_mean, 2),
        "Original Std Deviation": round(orig_std, 2),
        "Original Min Intensity": orig_min,
        "Original Max Intensity": orig_max,
        "CLAHE Mean Intensity": round(clahe_mean, 2),
        "CLAHE Std Deviation": round(clahe_std, 2),
        "CLAHE Min Intensity": clahe_min,
        "CLAHE Max Intensity": clahe_max,
        "Contrast Improvement (%)": round(contrast_improvement, 2)
    }

    return clahe_image, info

def create_comparison_image(image, clahe_image, save_path, output_format, quality):
    """Create a side-by-side comparison of original and CLAHE processed images."""
    # Get dimensions
    h, w = image.shape[:2]
    
    # Create a new image with twice the width
    comparison = np.zeros((h, w * 2), dtype=np.uint8)
    
    # Place original image on the left
    comparison[:, :w] = image
    
    # Place CLAHE image on the right
    comparison[:, w:] = clahe_image
    
    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    text_color = 255  # White
    
    # Add "Original" label
    cv2.putText(comparison, "Original", (10, 30), font, font_scale, text_color, font_thickness)
    
    # Add "CLAHE" label
    cv2.putText(comparison, "CLAHE", (w + 10, 30), font, font_scale, text_color, font_thickness)
    
    # Save comparison image
    try:
        if output_format.lower() == 'jpg' or output_format.lower() == 'jpeg':
            cv2.imwrite(save_path, comparison, [cv2.IMWRITE_JPEG_QUALITY, quality])
        elif output_format.lower() == 'png':
            cv2.imwrite(save_path, comparison, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        else:
            cv2.imwrite(save_path, comparison)
    except Exception as e:
        raise RuntimeError(f"Error saving comparison image: {str(e)}")

def create_histogram_plot(image, clahe_image, save_path):
    """Create histogram comparison plot."""
    plt.figure(figsize=(12, 8))
    
    # Create a 2x2 grid of subplots
    gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    
    # Original image
    ax1 = plt.subplot(gs[0, 0])
    ax1.imshow(image, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # CLAHE image
    ax2 = plt.subplot(gs[0, 1])
    ax2.imshow(clahe_image, cmap='gray')
    ax2.set_title('CLAHE Processed Image')
    ax2.axis('off')
    
    # Original histogram
    ax3 = plt.subplot(gs[1, 0])
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    ax3.plot(hist, color='gray')
    ax3.set_xlim([0, 256])
    ax3.set_title('Original Histogram')
    ax3.set_xlabel('Pixel Value')
    ax3.set_ylabel('Frequency')
    
    # CLAHE histogram
    ax4 = plt.subplot(gs[1, 1])
    hist = cv2.calcHist([clahe_image], [0], None, [256], [0, 256])
    ax4.plot(hist, color='gray')
    ax4.set_xlim([0, 256])
    ax4.set_title('CLAHE Histogram')
    ax4.set_xlabel('Pixel Value')
    ax4.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_intensity_distribution(image, clahe_image, save_path):
    """Create intensity distribution plot."""
    plt.figure(figsize=(12, 6))
    
    # Create a 1x2 grid of subplots
    gs = GridSpec(1, 2, width_ratios=[1, 1])
    
    # Original intensity distribution
    ax1 = plt.subplot(gs[0])
    ax1.hist(image.flatten(), bins=50, color='gray', alpha=0.7)
    ax1.set_title('Original Intensity Distribution')
    ax1.set_xlabel('Pixel Value')
    ax1.set_ylabel('Frequency')
    
    # CLAHE intensity distribution
    ax2 = plt.subplot(gs[1])
    ax2.hist(clahe_image.flatten(), bins=50, color='gray', alpha=0.7)
    ax2.set_title('CLAHE Intensity Distribution')
    ax2.set_xlabel('Pixel Value')
    ax2.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_metrics_visualization(image, clahe_image, processing_time, clip_limit, tile_grid_size, save_path):
    """Create metrics visualization."""
    # Calculate metrics
    orig_mean = np.mean(image)
    orig_std = np.std(image)
    orig_min = np.min(image)
    orig_max = np.max(image)
    
    clahe_mean = np.mean(clahe_image)
    clahe_std = np.std(clahe_image)
    clahe_min = np.min(clahe_image)
    clahe_max = np.max(clahe_image)
    
    # Calculate contrast improvement
    contrast_improvement = ((clahe_std - orig_std) / orig_std) * 100 if orig_std > 0 else 0
    
    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Mean intensity comparison
    ax1 = axs[0, 0]
    ax1.bar(['Original', 'CLAHE'], [orig_mean, clahe_mean], color=['gray', 'darkgray'])
    ax1.set_title('Mean Intensity Comparison')
    ax1.set_ylabel('Mean Intensity')
    
    # Standard deviation comparison
    ax2 = axs[0, 1]
    ax2.bar(['Original', 'CLAHE'], [orig_std, clahe_std], color=['gray', 'darkgray'])
    ax2.set_title('Standard Deviation Comparison')
    ax2.set_ylabel('Standard Deviation')
    
    # Intensity range comparison
    ax3 = axs[1, 0]
    ax3.bar(['Original Min', 'CLAHE Min'], [orig_min, clahe_min], color=['lightgray', 'silver'])
    ax3.bar(['Original Max', 'CLAHE Max'], [orig_max, clahe_max], color=['gray', 'darkgray'])
    ax3.set_title('Intensity Range Comparison')
    ax3.set_ylabel('Pixel Value')
    
    # Processing metrics
    ax4 = axs[1, 1]
    metrics = ['Clip Limit', 'Tile Size', 'Time (ms)', 'Contrast Improv. (%)']
    values = [clip_limit, tile_grid_size[0] * tile_grid_size[1], processing_time * 1000, contrast_improvement]
    ax4.bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
    ax4.set_title('Processing Metrics')
    ax4.set_ylabel('Value')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 8:
        print("Usage: python CLAHE.py <input_image_path> <output_image_or_directory_path> [clip_limit] [tile_grid_size] [format] [quality] [--no-visualizations]")
        print("Example: python CLAHE.py input.jpg output/ 2.0 (8,8) png")
        print("Example: python CLAHE.py input.jpg output_clahe.jpg 3.0 (4,4) jpg 90")
        print("Use --no-visualizations to skip creating research visualizations")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # Get optional parameters
    clip_limit = 2.0  # Default clip limit
    tile_grid_size = (8, 8)  # Default tile grid size
    output_format = 'jpg'  # Default format
    quality = 95  # Default quality
    create_visualizations = True  # Default to create visualizations
    
    # Parse clip_limit if provided
    if len(sys.argv) >= 4 and sys.argv[3] != "--no-visualizations":
        try:
            clip_limit = float(sys.argv[3])
        except ValueError:
            print("Warning: Invalid clip_limit value, using default (2.0)")
            clip_limit = 2.0
    
    # Parse tile_grid_size if provided
    if len(sys.argv) >= 5 and sys.argv[4] != "--no-visualizations":
        try:
            # Parse tuple format like "(8,8)"
            tile_str = sys.argv[4].strip('()')
            tile_parts = tile_str.split(',')
            tile_grid_size = (int(tile_parts[0]), int(tile_parts[1]))
        except Exception:
            print("Warning: Invalid tile_grid_size value, using default ((8,8))")
            tile_grid_size = (8, 8)
    
    # Parse output format if provided
    if len(sys.argv) >= 6 and sys.argv[5] != "--no-visualizations":
        output_format = sys.argv[5]
        if output_format.startswith('.'):
            output_format = output_format[1:]  # Remove dot if present
    
    # Parse quality if provided
    if len(sys.argv) >= 7 and sys.argv[6] != "--no-visualizations":
        try:
            quality = int(sys.argv[6])
            quality = max(0, min(100, quality))  # Ensure quality is between 0-100
        except ValueError:
            print("Warning: Invalid quality value, using default (95)")
            quality = 95
    
    # Check for no-visualizations flag
    if "--no-visualizations" in sys.argv:
        create_visualizations = False

    try:
        clahe_img, metadata = apply_clahe(
            input_path, output_path, clip_limit, tile_grid_size, 
            output_format, quality, create_visualizations
        )

        print("CLAHE processing complete. Metadata:")
        for key, value in metadata.items():
            print(f"{key}: {value}")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
