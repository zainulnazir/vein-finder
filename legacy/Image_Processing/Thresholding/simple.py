import cv2
import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.gridspec import GridSpec

def simple_threshold(input_path, output_dir, threshold_value, output_format='jpg', quality=95, create_visualizations=True):
    """
    Apply simple thresholding to an image with comprehensive research visualizations.
    
    Args:
        input_path: Path to the input image
        output_dir: Directory to save the output image
        threshold_value: Threshold value (0-255)
        output_format: Output image format ('jpg', 'png', etc.)
        quality: JPEG quality (0-100), only applicable for JPEG
        create_visualizations: Whether to create research visualizations
    
    Returns:
        tuple: (thresholded_image, metadata_dict)
    """
    # Read the image in grayscale
    gray = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not load image at {input_path}")

    # Apply simple thresholding
    start_time = time.time()
    _, thresh_img = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    processing_time = time.time() - start_time

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save output with specified format
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    save_path = os.path.join(output_dir, f"{base_name}_simple_thresh.{output_format}")
    
    try:
        if output_format.lower() == 'jpg' or output_format.lower() == 'jpeg':
            cv2.imwrite(save_path, thresh_img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        elif output_format.lower() == 'png':
            cv2.imwrite(save_path, thresh_img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        else:
            cv2.imwrite(save_path, thresh_img)
    except Exception as e:
        raise RuntimeError(f"Error saving thresholded image: {str(e)}")

    # Initialize visualization paths
    comparison_path = None
    histogram_path = None
    metrics_path = None
    region_path = None
    edge_path = None

    # Create research visualizations if requested
    if create_visualizations:
        # Create comparison image
        comparison_path = os.path.join(output_dir, f"{base_name}_comparison.{output_format}")
        create_comparison_image(gray, thresh_img, comparison_path, output_format, quality)
        
        # Create histogram analysis
        histogram_path = os.path.join(output_dir, f"{base_name}_histogram.png")
        create_histogram_analysis(gray, thresh_img, threshold_value, histogram_path)
        
        # Create metrics visualization
        metrics_path = os.path.join(output_dir, f"{base_name}_metrics.png")
        create_metrics_visualization(gray, thresh_img, threshold_value, processing_time, metrics_path)
        
        # Create region analysis
        region_path = os.path.join(output_dir, f"{base_name}_region_analysis.png")
        create_region_analysis(gray, thresh_img, threshold_value, region_path)
        
        # Create edge analysis
        edge_path = os.path.join(output_dir, f"{base_name}_edge_analysis.png")
        create_edge_analysis(gray, thresh_img, edge_path)

    # Calculate statistics
    total_pixels = gray.size
    foreground_pixels = np.count_nonzero(thresh_img)
    background_pixels = total_pixels - foreground_pixels
    
    # Calculate mean intensities
    foreground_mask = thresh_img > 0
    background_mask = thresh_img == 0
    
    foreground_mean = np.mean(gray[foreground_mask]) if np.any(foreground_mask) else 0
    background_mean = np.mean(gray[background_mask]) if np.any(background_mask) else 0
    
    # Calculate contrast measures
    contrast = abs(foreground_mean - background_mean)
    
    # Calculate Otsu's threshold for comparison
    otsu_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Get file sizes
    input_stats = os.stat(input_path)
    output_stats = os.stat(save_path)
    
    # Gather metadata
    height, width = gray.shape
    info = {
        "Input Image Path": input_path,
        "Output Image Path": save_path,
        "Comparison Image Path": comparison_path,
        "Histogram Analysis Path": histogram_path,
        "Metrics Visualization Path": metrics_path,
        "Region Analysis Path": region_path,
        "Edge Analysis Path": edge_path,
        "Image Dimensions": (height, width),
        "Threshold Value": threshold_value,
        "Otsu's Threshold": round(otsu_thresh, 2),
        "Total Pixels": total_pixels,
        "Foreground Pixels": foreground_pixels,
        "Background Pixels": background_pixels,
        "Foreground Percentage": round((foreground_pixels / total_pixels) * 100, 2),
        "Background Percentage": round((background_pixels / total_pixels) * 100, 2),
        "Foreground Mean Intensity": round(foreground_mean, 2),
        "Background Mean Intensity": round(background_mean, 2),
        "Contrast": round(contrast, 2),
        "Input File Size (KB)": round(input_stats.st_size / 1024, 2),
        "Output File Size (KB)": round(output_stats.st_size / 1024, 2),
        "File Size Change (%)": round((output_stats.st_size / input_stats.st_size - 1) * 100, 2),
        "Processing Time (ms)": round(processing_time * 1000, 2),
        "Processing Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    return thresh_img, info

def create_comparison_image(gray, thresh_img, save_path, output_format, quality):
    """Create a side-by-side comparison of original and thresholded images."""
    # Get dimensions
    h, w = gray.shape
    
    # Create a new image with twice the width
    comparison = np.zeros((h, w * 2), dtype=np.uint8)
    
    # Place original image on the left
    comparison[:, :w] = gray
    
    # Place thresholded image on the right
    comparison[:, w:] = thresh_img
    
    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    text_color = 255  # White
    
    # Add "Original" label
    cv2.putText(comparison, "Original", (10, 30), font, font_scale, text_color, font_thickness)
    
    # Add "Thresholded" label
    cv2.putText(comparison, "Thresholded", (w + 10, 30), font, font_scale, text_color, font_thickness)
    
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

def create_histogram_analysis(gray, thresh_img, threshold_value, save_path):
    """Create histogram analysis with threshold marked."""
    plt.figure(figsize=(12, 8))
    
    # Create a 2x2 grid of subplots
    gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    
    # Original image
    ax1 = plt.subplot(gs[0, 0])
    ax1.imshow(gray, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Thresholded image
    ax2 = plt.subplot(gs[0, 1])
    ax2.imshow(thresh_img, cmap='gray')
    ax2.set_title('Thresholded Image')
    ax2.axis('off')
    
    # Histogram with threshold marked
    ax3 = plt.subplot(gs[1, :])
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    ax3.plot(hist, color='gray')
    ax3.axvline(x=threshold_value, color='red', linestyle='--', linewidth=2, label=f'Threshold = {threshold_value}')
    
    # Calculate Otsu's threshold for comparison
    otsu_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ax3.axvline(x=otsu_thresh, color='green', linestyle='--', linewidth=2, label=f'Otsu\'s Threshold = {round(otsu_thresh, 2)}')
    
    ax3.set_xlim([0, 256])
    ax3.set_title('Histogram with Threshold Values')
    ax3.set_xlabel('Pixel Value')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_metrics_visualization(gray, thresh_img, threshold_value, processing_time, save_path):
    """Create metrics visualization."""
    # Calculate statistics
    total_pixels = gray.size
    foreground_pixels = np.count_nonzero(thresh_img)
    background_pixels = total_pixels - foreground_pixels
    
    # Calculate mean intensities
    foreground_mask = thresh_img > 0
    background_mask = thresh_img == 0
    
    foreground_mean = np.mean(gray[foreground_mask]) if np.any(foreground_mask) else 0
    background_mean = np.mean(gray[background_mask]) if np.any(background_mask) else 0
    
    # Calculate contrast measures
    contrast = abs(foreground_mean - background_mean)
    
    # Calculate Otsu's threshold for comparison
    otsu_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    plt.figure(figsize=(12, 8))
    
    # Create a 2x2 grid of subplots
    gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    
    # Pixel distribution
    ax1 = plt.subplot(gs[0, 0])
    ax1.bar(['Foreground', 'Background'], [foreground_pixels, background_pixels], color=['white', 'black'])
    ax1.set_title('Pixel Distribution')
    ax1.set_ylabel('Number of Pixels')
    
    # Mean intensities
    ax2 = plt.subplot(gs[0, 1])
    ax2.bar(['Foreground', 'Background'], [foreground_mean, background_mean], color=['white', 'black'])
    ax2.set_title('Mean Intensity by Region')
    ax2.set_ylabel('Mean Intensity')
    
    # Threshold comparison
    ax3 = plt.subplot(gs[1, 0])
    ax3.bar(['Used Threshold', 'Otsu\'s Threshold'], [threshold_value, otsu_thresh], color=['red', 'green'])
    ax3.set_title('Threshold Comparison')
    ax3.set_ylabel('Threshold Value')
    
    # Processing metrics
    ax4 = plt.subplot(gs[1, 1])
    metrics = ['Processing Time (ms)', 'Contrast']
    values = [processing_time * 1000, contrast]
    ax4.bar(metrics, values, color=['blue', 'purple'])
    ax4.set_title('Processing Metrics')
    ax4.set_ylabel('Value')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_region_analysis(gray, thresh_img, threshold_value, save_path):
    """Create region analysis visualization."""
    # Create masks for foreground and background
    foreground_mask = thresh_img > 0
    background_mask = thresh_img == 0
    
    # Calculate mean intensities
    foreground_mean = np.mean(gray[foreground_mask]) if np.any(foreground_mask) else 0
    background_mean = np.mean(gray[background_mask]) if np.any(background_mask) else 0
    
    # Create images showing only foreground and background regions
    foreground_img = gray.copy()
    foreground_img[background_mask] = 0
    
    background_img = gray.copy()
    background_img[foreground_mask] = 0
    
    plt.figure(figsize=(12, 8))
    
    # Create a 2x2 grid of subplots
    gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    
    # Original image
    ax1 = plt.subplot(gs[0, 0])
    ax1.imshow(gray, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Thresholded image
    ax2 = plt.subplot(gs[0, 1])
    ax2.imshow(thresh_img, cmap='gray')
    ax2.set_title('Thresholded Image')
    ax2.axis('off')
    
    # Foreground regions
    ax3 = plt.subplot(gs[1, 0])
    ax3.imshow(foreground_img, cmap='gray')
    ax3.set_title(f'Foreground Regions (Mean: {round(foreground_mean, 2)})')
    ax3.axis('off')
    
    # Background regions
    ax4 = plt.subplot(gs[1, 1])
    ax4.imshow(background_img, cmap='gray')
    ax4.set_title(f'Background Regions (Mean: {round(background_mean, 2)})')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_edge_analysis(gray, thresh_img, save_path):
    """Create edge analysis visualization."""
    # Apply Canny edge detection to original image
    edges_original = cv2.Canny(gray, 100, 200)
    
    # Apply Canny edge detection to thresholded image
    edges_thresh = cv2.Canny(thresh_img, 100, 200)
    
    # Calculate edge preservation metric
    intersection = np.logical_and(edges_original > 0, edges_thresh > 0)
    union = np.logical_or(edges_original > 0, edges_thresh > 0)
    
    if np.sum(union) > 0:
        jaccard = np.sum(intersection) / np.sum(union)
    else:
        jaccard = 1.0  # Both have no edges
    
    plt.figure(figsize=(12, 8))
    
    # Create a 2x2 grid of subplots
    gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    
    # Original edges
    ax1 = plt.subplot(gs[0, 0])
    ax1.imshow(edges_original, cmap='gray')
    ax1.set_title('Original Image Edges')
    ax1.axis('off')
    
    # Thresholded edges
    ax2 = plt.subplot(gs[0, 1])
    ax2.imshow(edges_thresh, cmap='gray')
    ax2.set_title('Thresholded Image Edges')
    ax2.axis('off')
    
    # Edge overlap
    ax3 = plt.subplot(gs[1, 0])
    overlap = np.zeros_like(edges_original)
    overlap[intersection] = 255  # White for overlapping edges
    overlap[edges_original > 0] = 100  # Gray for edges only in original
    overlap[edges_thresh > 0] = 50  # Dark gray for edges only in thresholded
    ax3.imshow(overlap, cmap='gray')
    ax3.set_title('Edge Overlap Analysis')
    ax3.axis('off')
    
    # Edge preservation metric
    ax4 = plt.subplot(gs[1, 1])
    metrics = ['Edge Preservation (Jaccard)']
    values = [jaccard]
    ax4.bar(metrics, values, color='purple')
    ax4.set_ylim([0, 1])
    ax4.set_title('Edge Preservation Metric')
    ax4.set_ylabel('Value')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) < 4 or len(sys.argv) > 8:
        print("Usage: python simple.py <input_image> <output_dir> <threshold_value> [format] [quality] [--no-visualizations]")
        print("Example: python simple.py input.jpg output/ 127")
        print("Example: python simple.py input.jpg output/ 150 png 90")
        print("Use --no-visualizations to skip creating research visualizations")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2]
    threshold_value = int(sys.argv[3])
    
    # Get optional parameters
    output_format = 'jpg'  # Default format
    quality = 95           # Default quality
    create_visualizations = True  # Default to create visualizations
    
    # Parse output format if provided
    if len(sys.argv) >= 5 and sys.argv[4] != "--no-visualizations":
        output_format = sys.argv[4]
        if output_format.startswith('.'):
            output_format = output_format[1:]  # Remove dot if present
    
    # Parse quality if provided
    if len(sys.argv) >= 6 and sys.argv[5] != "--no-visualizations":
        try:
            quality = int(sys.argv[5])
            quality = max(0, min(100, quality))  # Ensure quality is between 0-100
        except ValueError:
            print("Warning: Invalid quality value, using default (95)")
            quality = 95
    
    # Check for no-visualizations flag
    if "--no-visualizations" in sys.argv:
        create_visualizations = False

    try:
        thresh_img, metadata = simple_threshold(
            input_path, output_dir, threshold_value, 
            output_format, quality, create_visualizations
        )

        print("Thresholding complete. Metadata:")
        for key, value in metadata.items():
            print(f"{key}: {value}")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
