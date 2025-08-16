import cv2
import sys
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.gridspec import GridSpec

def adaptive_threshold_vein(input_path, output_dir, block_size=None, C=None, output_format='jpg', quality=95, create_visualizations=True):
    """
    Apply adaptive thresholding to vein images with comprehensive research visualizations.
    
    Args:
        input_path: Path to the input image
        output_dir: Directory to save the output image
        block_size: Block size for adaptive thresholding (optional, auto-determined if None)
        C: Constant subtracted from the calculated threshold (optional, auto-determined if None)
        output_format: Output image format ('jpg', 'png', etc.)
        quality: JPEG quality (0-100), only applicable for JPEG
        create_visualizations: Whether to create research visualizations
    
    Returns:
        tuple: (thresholded_image, metadata_dict)
    """
    # Read the image
    gray = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read image at {input_path}")
    
    # Store original image for comparison
    original = gray.copy()
    
    # Apply intensity mask to exclude shadows
    # Use a combination of global thresholding and local variance to detect shadows
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Calculate local variance to identify regions with low detail (potential shadows)
    mean = cv2.blur(blurred, (15, 15))
    mean_sq = cv2.blur(blurred**2, (15, 15))
    variance = mean_sq - mean**2
    variance_norm = cv2.normalize(variance, None, 0, 255, cv2.NORM_MINMAX)
    
    # Adaptive threshold for variance based on image characteristics
    var_mean = np.mean(variance_norm)
    var_threshold = max(20, min(40, var_mean * 0.8))
    
    # Threshold to identify shadow regions
    _, intensity_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    intensity_thresh = cv2.bitwise_not(intensity_thresh)  # Invert to get dark regions
    _, variance_thresh = cv2.threshold(variance_norm, var_threshold, 255, cv2.THRESH_BINARY)
    
    # Combine thresholds to identify shadows
    shadow_mask = cv2.bitwise_and(intensity_thresh, variance_thresh)
    kernel = np.ones((5,5), np.uint8)
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)
    intensity_mask = cv2.bitwise_not(shadow_mask)
    
    # Apply the mask to the original image
    masked_image = gray.copy()
    masked_image[intensity_mask == 0] = 255  # Set shadow regions to white
    
    # Determine block size
    if block_size is None:
        height, width = gray.shape
        min_dim = min(height, width)
        
        # Set block size based on image resolution
        if min_dim < 500:
            block_size = 15  # Fixed small block size for low-res images
        elif min_dim > 1000:
            block_size = 31  # Fixed medium block size for high-res images
        else:
            # For medium resolution, scale with image size
            block_size = int(min_dim * 0.03)  # 3% of the smaller dimension
            if block_size % 2 == 0:
                block_size += 1
            # Ensure block size is within reasonable bounds
            block_size = max(15, min(block_size, 31))
    else:
        # Ensure block size is odd and greater than 1
        if block_size % 2 == 0:
            block_size += 1
        if block_size <= 1:
            block_size = 3
    
    # Determine C parameter
    if C is None:
        # Automatically determine C based on image contrast
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        
        # Set C based on the contrast (std/mean ratio) but within reasonable bounds
        contrast_ratio = std_intensity / mean_intensity
        if contrast_ratio < 0.1:  # Low contrast
            C = 2
        elif contrast_ratio > 0.3:  # High contrast
            C = 10
        else:  # Medium contrast
            C = int(2 + (contrast_ratio - 0.1) * 40)  # Scale between 2 and 10
            C = max(2, min(C, 10))
    
    # Apply Gaussian adaptive thresholding
    start_time = time.time()
    thresh = cv2.adaptiveThreshold(
        masked_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,  # Use THRESH_BINARY_INV to highlight dark veins
        block_size,
        C
    )
    processing_time = time.time() - start_time
    
    # Apply the intensity mask again to ensure shadow regions remain white
    thresh[intensity_mask == 0] = 0
    
    # Apply morphological operations to clean up the result
    # Small kernel to remove noise while preserving vein structures
    kernel = np.ones((2,2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize visualization paths
    comparison_path = None
    histogram_path = None
    region_analysis_path = None
    parameter_analysis_path = None
    metrics_path = None
    edge_analysis_path = None
    
    # Save output
    if block_size is None or C is None:
        filename = os.path.splitext(os.path.basename(input_path))[0] + f"_adaptive_vein_auto.{output_format}"
    else:
        filename = os.path.splitext(os.path.basename(input_path))[0] + f"_adaptive_vein_{block_size}_{C}.{output_format}"
    save_path = os.path.join(output_dir, filename)
    
    try:
        if output_format.lower() == 'jpg' or output_format.lower() == 'jpeg':
            cv2.imwrite(save_path, thresh, [cv2.IMWRITE_JPEG_QUALITY, quality])
        elif output_format.lower() == 'png':
            cv2.imwrite(save_path, thresh, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        else:
            cv2.imwrite(save_path, thresh)
    except Exception as e:
        raise RuntimeError(f"Error saving thresholded image: {str(e)}")
    
    # Create research visualizations if requested
    if create_visualizations:
        # Create comparison image
        comparison_path = save_path.replace(f"_adaptive_vein_auto.{output_format}", f"_comparison.{output_format}")
        comparison_path = comparison_path.replace(f"_adaptive_vein_{block_size}_{C}.{output_format}", f"_comparison.{output_format}")
        create_comparison_image(original, thresh, comparison_path, output_format, quality)
        
        # Create histogram analysis
        histogram_path = save_path.replace(f"_adaptive_vein_auto.{output_format}", f"_histogram.png")
        histogram_path = histogram_path.replace(f"_adaptive_vein_{block_size}_{C}.{output_format}", f"_histogram.png")
        create_histogram_analysis(original, thresh, histogram_path)
        
        # Create region analysis
        region_analysis_path = save_path.replace(f"_adaptive_vein_auto.{output_format}", f"_region_analysis.png")
        region_analysis_path = region_analysis_path.replace(f"_adaptive_vein_{block_size}_{C}.{output_format}", f"_region_analysis.png")
        create_region_analysis(original, thresh, intensity_mask, region_analysis_path)
        
        # Create parameter analysis
        parameter_analysis_path = save_path.replace(f"_adaptive_vein_auto.{output_format}", f"_parameter_analysis.png")
        parameter_analysis_path = parameter_analysis_path.replace(f"_adaptive_vein_{block_size}_{C}.{output_format}", f"_parameter_analysis.png")
        create_parameter_analysis(original, block_size, C, parameter_analysis_path)
        
        # Create metrics visualization
        metrics_path = save_path.replace(f"_adaptive_vein_auto.{output_format}", f"_metrics.png")
        metrics_path = metrics_path.replace(f"_adaptive_vein_{block_size}_{C}.{output_format}", f"_metrics.png")
        create_metrics_visualization(original, thresh, processing_time, block_size, C, metrics_path)
        
        # Create edge analysis
        edge_analysis_path = save_path.replace(f"_adaptive_vein_auto.{output_format}", f"_edge_analysis.png")
        edge_analysis_path = edge_analysis_path.replace(f"_adaptive_vein_{block_size}_{C}.{output_format}", f"_edge_analysis.png")
        create_edge_analysis(original, thresh, edge_analysis_path)
    
    # Calculate statistics
    total_pixels = original.size
    vein_pixels = np.count_nonzero(thresh)
    background_pixels = total_pixels - vein_pixels
    
    # Calculate mean intensities
    vein_mask = thresh > 0
    background_mask = thresh == 0
    
    vein_mean = np.mean(original[vein_mask]) if np.any(vein_mask) else 0
    background_mean = np.mean(original[background_mask]) if np.any(background_mask) else 0
    
    # Calculate contrast measures
    contrast = abs(vein_mean - background_mean)
    
    # Calculate local contrast metrics
    # Divide image into blocks and calculate contrast in each block
    h, w = original.shape
    block_h, block_w = h // 10, w // 10  # 10x10 grid
    local_contrasts = []
    
    for i in range(0, h, block_h):
        for j in range(0, w, block_w):
            block = original[i:i+block_h, j:j+block_w]
            if block.size > 0:
                local_contrasts.append(np.std(block))
    
    avg_local_contrast = np.mean(local_contrasts) if local_contrasts else 0
    std_local_contrast = np.std(local_contrasts) if local_contrasts else 0
    
    # Get file sizes
    input_stats = os.stat(input_path)
    output_stats = os.stat(save_path)
    
    # Gather metadata
    info = {
        "Input Image Path": input_path,
        "Output Image Path": save_path,
        "Comparison Image Path": comparison_path,
        "Histogram Analysis Path": histogram_path,
        "Region Analysis Path": region_analysis_path,
        "Parameter Analysis Path": parameter_analysis_path,
        "Metrics Visualization Path": metrics_path,
        "Edge Analysis Path": edge_analysis_path,
        "Image Dimensions": original.shape,
        "Block Size": block_size,
        "C Value": C,
        "Total Pixels": total_pixels,
        "Vein Pixels": vein_pixels,
        "Background Pixels": background_pixels,
        "Vein Percentage": round((vein_pixels / total_pixels) * 100, 2),
        "Background Percentage": round((background_pixels / total_pixels) * 100, 2),
        "Vein Mean Intensity": round(vein_mean, 2),
        "Background Mean Intensity": round(background_mean, 2),
        "Contrast": round(contrast, 2),
        "Average Local Contrast": round(avg_local_contrast, 2),
        "Std Local Contrast": round(std_local_contrast, 2),
        "Input File Size (KB)": round(input_stats.st_size / 1024, 2),
        "Output File Size (KB)": round(output_stats.st_size / 1024, 2),
        "File Size Change (%)": round((output_stats.st_size / input_stats.st_size - 1) * 100, 2),
        "Processing Time (ms)": round(processing_time * 1000, 2),
        "Processing Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    return thresh, info

def create_comparison_image(original, thresh, save_path, output_format, quality):
    """Create a side-by-side comparison of original and thresholded images."""
    # Get dimensions
    h, w = original.shape
    
    # Create a new image with twice the width
    comparison = np.zeros((h, w * 2), dtype=np.uint8)
    
    # Place original image on the left
    comparison[:, :w] = original
    
    # Place thresholded image on the right
    comparison[:, w:] = thresh
    
    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    text_color = 255  # White
    
    # Add "Original" label
    cv2.putText(comparison, "Original", (10, 30), font, font_scale, text_color, font_thickness)
    
    # Add "Adaptive Threshold" label
    cv2.putText(comparison, "Adaptive Threshold", (w + 10, 30), font, font_scale, text_color, font_thickness)
    
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

def create_histogram_analysis(original, thresh, save_path):
    """Create histogram analysis comparing original and thresholded images."""
    plt.figure(figsize=(12, 8))
    
    # Create a 2x2 grid of subplots
    gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    
    # Original image
    ax1 = plt.subplot(gs[0, 0])
    ax1.imshow(original, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Thresholded image
    ax2 = plt.subplot(gs[0, 1])
    ax2.imshow(thresh, cmap='gray')
    ax2.set_title('Adaptive Thresholded Image')
    ax2.axis('off')
    
    # Original histogram
    ax3 = plt.subplot(gs[1, 0])
    hist = cv2.calcHist([original], [0], None, [256], [0, 256])
    ax3.plot(hist, color='gray')
    ax3.set_xlim([0, 256])
    ax3.set_title('Original Histogram')
    ax3.set_xlabel('Pixel Value')
    ax3.set_ylabel('Frequency')
    
    # Thresholded histogram
    ax4 = plt.subplot(gs[1, 1])
    hist_thresh = cv2.calcHist([thresh], [0], None, [256], [0, 256])
    ax4.plot(hist_thresh, color='red')
    ax4.set_xlim([0, 256])
    ax4.set_title('Thresholded Histogram')
    ax4.set_xlabel('Pixel Value')
    ax4.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_region_analysis(original, thresh, intensity_mask, save_path):
    """Create region analysis showing vein and background regions."""
    # Create masks for vein and background
    vein_mask = thresh > 0
    background_mask = thresh == 0
    
    # Calculate mean intensities
    vein_mean = np.mean(original[vein_mask]) if np.any(vein_mask) else 0
    background_mean = np.mean(original[background_mask]) if np.any(background_mask) else 0
    
    # Create images showing only vein and background regions
    vein_img = original.copy()
    vein_img[background_mask] = 0
    
    background_img = original.copy()
    background_img[vein_mask] = 0
    
    # Create shadow mask visualization
    shadow_img = original.copy()
    shadow_img[intensity_mask == 0] = 0
    
    plt.figure(figsize=(16, 12))
    
    # Create a 2x3 grid of subplots
    gs = GridSpec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1])
    
    # Original image
    ax1 = plt.subplot(gs[0, 0])
    ax1.imshow(original, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Thresholded image
    ax2 = plt.subplot(gs[0, 1])
    ax2.imshow(thresh, cmap='gray')
    ax2.set_title('Adaptive Thresholded Image')
    ax2.axis('off')
    
    # Shadow mask
    ax3 = plt.subplot(gs[0, 2])
    ax3.imshow(shadow_img, cmap='gray')
    ax3.set_title('Shadow Mask')
    ax3.axis('off')
    
    # Vein regions
    ax4 = plt.subplot(gs[1, 0])
    ax4.imshow(vein_img, cmap='gray')
    ax4.set_title(f'Vein Regions (Mean: {round(vein_mean, 2)})')
    ax4.axis('off')
    
    # Background regions
    ax5 = plt.subplot(gs[1, 1])
    ax5.imshow(background_img, cmap='gray')
    ax5.set_title(f'Background Regions (Mean: {round(background_mean, 2)})')
    ax5.axis('off')
    
    # Intensity distribution
    ax6 = plt.subplot(gs[1, 2])
    ax6.hist(original[vein_mask].flatten(), bins=50, alpha=0.5, color='red', label='Vein')
    ax6.hist(original[background_mask].flatten(), bins=50, alpha=0.5, color='blue', label='Background')
    ax6.set_title('Intensity Distribution')
    ax6.set_xlabel('Pixel Value')
    ax6.set_ylabel('Frequency')
    ax6.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_parameter_analysis(original, block_size, C, save_path):
    """Create parameter analysis showing the effect of different parameters."""
    plt.figure(figsize=(16, 10))
    
    # Create a 2x3 grid of subplots
    gs = GridSpec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1])
    
    # Test different block sizes
    block_sizes = [5, 15, 31]
    for i, bs in enumerate(block_sizes):
        ax = plt.subplot(gs[0, i])
        thresh_bs = cv2.adaptiveThreshold(
            original,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            bs,
            C
        )
        ax.imshow(thresh_bs, cmap='gray')
        ax.set_title(f'Block Size: {bs}')
        ax.axis('off')
    
    # Test different C values
    C_values = [2, 5, 10]
    for i, c_val in enumerate(C_values):
        ax = plt.subplot(gs[1, i])
        thresh_c = cv2.adaptiveThreshold(
            original,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            c_val
        )
        ax.imshow(thresh_c, cmap='gray')
        ax.set_title(f'C Value: {c_val}')
        ax.axis('off')
    
    plt.suptitle(f'Parameter Analysis (Used: Block Size={block_size}, C={C})', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_metrics_visualization(original, thresh, processing_time, block_size, C, save_path):
    """Create metrics visualization."""
    # Calculate statistics
    total_pixels = original.size
    vein_pixels = np.count_nonzero(thresh)
    background_pixels = total_pixels - vein_pixels
    
    # Calculate mean intensities
    vein_mask = thresh > 0
    background_mask = thresh == 0
    
    vein_mean = np.mean(original[vein_mask]) if np.any(vein_mask) else 0
    background_mean = np.mean(original[background_mask]) if np.any(background_mask) else 0
    
    # Calculate contrast measures
    contrast = abs(vein_mean - background_mean)
    
    # Calculate local contrast metrics
    h, w = original.shape
    block_h, block_w = h // 10, w // 10  # 10x10 grid
    local_contrasts = []
    
    for i in range(0, h, block_h):
        for j in range(0, w, block_w):
            block = original[i:i+block_h, j:j+block_w]
            if block.size > 0:
                local_contrasts.append(np.std(block))
    
    avg_local_contrast = np.mean(local_contrasts) if local_contrasts else 0
    std_local_contrast = np.std(local_contrasts) if local_contrasts else 0
    
    plt.figure(figsize=(12, 10))
    
    # Create a 2x2 grid of subplots
    gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    
    # Pixel distribution
    ax1 = plt.subplot(gs[0, 0])
    ax1.bar(['Vein', 'Background'], [vein_pixels, background_pixels], color=['red', 'blue'])
    ax1.set_title('Pixel Distribution')
    ax1.set_ylabel('Number of Pixels')
    
    # Mean intensities
    ax2 = plt.subplot(gs[0, 1])
    ax2.bar(['Vein', 'Background'], [vein_mean, background_mean], color=['red', 'blue'])
    ax2.set_title('Mean Intensity by Region')
    ax2.set_ylabel('Mean Intensity')
    
    # Parameters and performance
    ax3 = plt.subplot(gs[1, 0])
    metrics = ['Block Size', 'C Value', 'Processing Time (ms)']
    values = [block_size, C, processing_time * 1000]
    ax3.bar(metrics, values, color=['green', 'orange', 'purple'])
    ax3.set_title('Parameters and Performance')
    ax3.set_ylabel('Value')
    
    # Contrast metrics
    ax4 = plt.subplot(gs[1, 1])
    contrast_metrics = ['Global Contrast', 'Avg Local Contrast', 'Std Local Contrast']
    contrast_values = [contrast, avg_local_contrast, std_local_contrast]
    ax4.bar(contrast_metrics, contrast_values, color=['cyan', 'magenta', 'yellow'])
    ax4.set_title('Contrast Metrics')
    ax4.set_ylabel('Value')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_edge_analysis(original, thresh, save_path):
    """Create edge analysis visualization."""
    # Apply Canny edge detection to original image
    edges_original = cv2.Canny(original, 100, 200)
    
    # Apply Canny edge detection to thresholded image
    edges_thresh = cv2.Canny(thresh, 100, 200)
    
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
    if len(sys.argv) not in [3, 5, 7, 8]:
        print("Usage: python adaptive.py <input_image> <output_dir> [block_size C] [format] [quality] [--no-visualizations]")
        print("If block_size and C are not provided, they will be determined automatically")
        print("Example: python adaptive.py input.jpg output/ 200 1")
        print("Example: python adaptive.py input.jpg output/ 200 1 png 90")
        print("Use --no-visualizations to skip creating research visualizations")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_dir = sys.argv[2]
    
    # Get optional parameters
    block_size = None
    C = None
    output_format = 'jpg'  # Default format
    quality = 95  # Default quality
    create_visualizations = True  # Default to create visualizations
    
    # Parse block_size and C if provided
    if len(sys.argv) >= 5 and sys.argv[3] != "--no-visualizations":
        block_size = int(sys.argv[3])
        C = int(sys.argv[4])
    
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
        thresh_img, metadata = adaptive_threshold_vein(
            input_path, output_dir, block_size, C,
            output_format, quality, create_visualizations
        )
        
        print("Adaptive thresholding complete. Metadata:")
        for key, value in metadata.items():
            print(f"{key}: {value}")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
