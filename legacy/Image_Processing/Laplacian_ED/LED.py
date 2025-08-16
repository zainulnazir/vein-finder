import cv2
import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.gridspec import GridSpec

def laplacian_edge_detection(input_path, output_dir, kernel_size=3, threshold=None, output_format='jpg', quality=95, create_visualizations=True):
    """
    Apply Laplacian edge detection with improved normalization and thresholding.
    
    Args:
        input_path: Path to the input image
        output_dir: Directory to save the output image
        kernel_size: Size of the Laplacian kernel (1, 3, 5, or 7)
        threshold: Threshold value for binary edge map (0-255, None for automatic)
        output_format: Output image format ('jpg', 'png', etc.)
        quality: JPEG quality (0-100), only applicable for JPEG
        create_visualizations: Whether to create research visualizations
    
    Returns:
        tuple: (edge_image, metadata_dict)
    """
    # Check if input file exists
    if not os.path.isfile(input_path):
        # List files in current directory to help user
        current_dir = os.getcwd()
        files = os.listdir(current_dir)
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        error_msg = f"Error: Could not read image at {input_path}\n"
        error_msg += f"Current directory: {current_dir}\n"
        if image_files:
            error_msg += f"Available image files: {', '.join(image_files)}"
        else:
            error_msg += "No image files found in current directory"
        
        raise FileNotFoundError(error_msg)
    
    # Read the image
    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {input_path}")
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        color_image = image.copy()
    else:
        gray = image.copy()
        color_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # Store original for comparison
    original = gray.copy()
    
    # Apply Gaussian blur to reduce noise (optional but recommended for Laplacian)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Apply Laplacian edge detection
    start_time = time.time()
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=kernel_size)
    
    # Normalize the Laplacian to 0-255 range for better visualization
    # First, get the absolute values
    laplacian_abs = np.absolute(laplacian)
    
    # Then normalize to 0-255
    laplacian_norm = cv2.normalize(laplacian_abs, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # Apply threshold to create a binary edge map
    if threshold is None:
        # Use Otsu's method to automatically determine the threshold
        _, edge_binary = cv2.threshold(laplacian_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # Use the provided threshold
        _, edge_binary = cv2.threshold(laplacian_norm, threshold, 255, cv2.THRESH_BINARY)
    
    processing_time = time.time() - start_time
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize visualization paths
    comparison_path = None
    histogram_path = None
    edge_strength_path = None
    metrics_path = None
    kernel_analysis_path = None
    edge_overlay_path = None
    
    # Save output
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    # Save normalized Laplacian result
    norm_path = os.path.join(output_dir, f"{base_name}_laplacian_norm.{output_format}")
    try:
        if output_format.lower() == 'jpg' or output_format.lower() == 'jpeg':
            cv2.imwrite(norm_path, laplacian_norm, [cv2.IMWRITE_JPEG_QUALITY, quality])
        elif output_format.lower() == 'png':
            cv2.imwrite(norm_path, laplacian_norm, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        else:
            cv2.imwrite(norm_path, laplacian_norm)
    except Exception as e:
        raise RuntimeError(f"Error saving normalized Laplacian image: {str(e)}")
    
    # Save binary edge map
    binary_path = os.path.join(output_dir, f"{base_name}_laplacian_binary.{output_format}")
    try:
        if output_format.lower() == 'jpg' or output_format.lower() == 'jpeg':
            cv2.imwrite(binary_path, edge_binary, [cv2.IMWRITE_JPEG_QUALITY, quality])
        elif output_format.lower() == 'png':
            cv2.imwrite(binary_path, edge_binary, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        else:
            cv2.imwrite(binary_path, edge_binary)
    except Exception as e:
        raise RuntimeError(f"Error saving binary edge image: {str(e)}")
    
    # Create research visualizations if requested
    if create_visualizations:
        # Create comparison image
        comparison_path = os.path.join(output_dir, f"{base_name}_comparison.{output_format}")
        create_comparison_image(color_image, laplacian_norm, comparison_path, output_format, quality)
        
        # Create histogram analysis
        histogram_path = os.path.join(output_dir, f"{base_name}_histogram.png")
        create_histogram_analysis(original, laplacian_norm, histogram_path)
        
        # Create edge strength distribution
        edge_strength_path = os.path.join(output_dir, f"{base_name}_edge_strength.png")
        create_edge_strength_distribution(laplacian_norm, edge_strength_path)
        
        # Create metrics visualization
        metrics_path = os.path.join(output_dir, f"{base_name}_metrics.png")
        create_metrics_visualization(original, laplacian_norm, edge_binary, processing_time, kernel_size, threshold, metrics_path)
        
        # Create kernel analysis
        kernel_analysis_path = os.path.join(output_dir, f"{base_name}_kernel_analysis.png")
        create_kernel_analysis(original, kernel_analysis_path)
        
        # Create edge overlay
        edge_overlay_path = os.path.join(output_dir, f"{base_name}_edge_overlay.{output_format}")
        create_edge_overlay(color_image, edge_binary, edge_overlay_path, output_format, quality)
    
    # Calculate statistics
    total_pixels = original.size
    edge_pixels = np.count_nonzero(edge_binary)
    non_edge_pixels = total_pixels - edge_pixels
    edge_density = edge_pixels / total_pixels
    
    # Calculate edge strength statistics
    edge_strength_mean = np.mean(laplacian_norm)
    edge_strength_std = np.std(laplacian_norm)
    edge_strength_min = np.min(laplacian_norm)
    edge_strength_max = np.max(laplacian_norm)
    
    # Calculate edge strength in non-zero regions only
    non_zero_laplacian = laplacian_norm[laplacian_norm > 0]
    if non_zero_laplacian.size > 0:
        non_zero_mean = np.mean(non_zero_laplacian)
        non_zero_std = np.std(non_zero_laplacian)
    else:
        non_zero_mean = 0
        non_zero_std = 0
    
    # Get file sizes
    input_stats = os.stat(input_path)
    norm_stats = os.stat(norm_path)
    binary_stats = os.stat(binary_path)
    
    # Gather metadata
    height, width = original.shape
    info = {
        "Input Image Path": input_path,
        "Laplacian Normalized Path": norm_path,
        "Laplacian Binary Path": binary_path,
        "Comparison Image Path": comparison_path,
        "Histogram Analysis Path": histogram_path,
        "Edge Strength Distribution Path": edge_strength_path,
        "Metrics Visualization Path": metrics_path,
        "Kernel Analysis Path": kernel_analysis_path,
        "Edge Overlay Path": edge_overlay_path,
        "Image Dimensions": (height, width),
        "Kernel Size": kernel_size,
        "Threshold Used": threshold if threshold is not None else "Otsu's",
        "Total Pixels": total_pixels,
        "Edge Pixels": edge_pixels,
        "Non-Edge Pixels": non_edge_pixels,
        "Edge Density": round(edge_density, 4),
        "Edge Strength Mean": round(edge_strength_mean, 2),
        "Edge Strength Std": round(edge_strength_std, 2),
        "Edge Strength Min": edge_strength_min,
        "Edge Strength Max": edge_strength_max,
        "Non-Zero Edge Strength Mean": round(non_zero_mean, 2),
        "Non-Zero Edge Strength Std": round(non_zero_std, 2),
        "Input File Size (KB)": round(input_stats.st_size / 1024, 2),
        "Laplacian Normalized File Size (KB)": round(norm_stats.st_size / 1024, 2),
        "Laplacian Binary File Size (KB)": round(binary_stats.st_size / 1024, 2),
        "Processing Time (ms)": round(processing_time * 1000, 2),
        "Processing Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    return laplacian_norm, info

def create_comparison_image(color_image, laplacian_norm, save_path, output_format, quality):
    """Create a side-by-side comparison of original and edge-detected images."""
    # Get dimensions
    h, w = color_image.shape[:2]
    
    # Create a new image with twice the width
    comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
    
    # Place original image on the left
    comparison[:, :w] = color_image
    
    # Convert Laplacian to BGR for display
    laplacian_bgr = cv2.cvtColor(laplacian_norm, cv2.COLOR_GRAY2BGR)
    
    # Place Laplacian image on the right
    comparison[:, w:] = laplacian_bgr
    
    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    text_color = (255, 255, 255)  # White
    
    # Add "Original" label
    cv2.putText(comparison, "Original", (10, 30), font, font_scale, text_color, font_thickness)
    
    # Add "Laplacian Edges" label
    cv2.putText(comparison, "Laplacian Edges", (w + 10, 30), font, font_scale, text_color, font_thickness)
    
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

def create_histogram_analysis(original, laplacian_norm, save_path):
    """Create histogram analysis comparing original and edge-detected images."""
    plt.figure(figsize=(12, 8))
    
    # Create a 2x2 grid of subplots
    gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    
    # Original image
    ax1 = plt.subplot(gs[0, 0])
    ax1.imshow(original, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Laplacian image
    ax2 = plt.subplot(gs[0, 1])
    ax2.imshow(laplacian_norm, cmap='gray')
    ax2.set_title('Laplacian Edge Detection')
    ax2.axis('off')
    
    # Original histogram
    ax3 = plt.subplot(gs[1, 0])
    hist = cv2.calcHist([original], [0], None, [256], [0, 256])
    ax3.plot(hist, color='gray')
    ax3.set_xlim([0, 256])
    ax3.set_title('Original Histogram')
    ax3.set_xlabel('Pixel Value')
    ax3.set_ylabel('Frequency')
    
    # Laplacian histogram
    ax4 = plt.subplot(gs[1, 1])
    hist_lap = cv2.calcHist([laplacian_norm], [0], None, [256], [0, 256])
    ax4.plot(hist_lap, color='red')
    ax4.set_xlim([0, 256])
    ax4.set_title('Laplacian Histogram')
    ax4.set_xlabel('Pixel Value')
    ax4.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_edge_strength_distribution(laplacian_norm, save_path):
    """Create edge strength distribution visualization."""
    # Flatten the Laplacian for histogram
    laplacian_flat = laplacian_norm.flatten()
    
    # Remove zeros for better visualization of edge strengths
    non_zero_laplacian = laplacian_flat[laplacian_flat > 0]
    
    plt.figure(figsize=(12, 6))
    
    # Create a 1x2 grid of subplots
    gs = GridSpec(1, 2, width_ratios=[1, 1])
    
    # Full Laplacian distribution
    ax1 = plt.subplot(gs[0])
    ax1.hist(laplacian_flat, bins=100, color='blue', alpha=0.7)
    ax1.set_title('Full Laplacian Distribution')
    ax1.set_xlabel('Laplacian Value')
    ax1.set_ylabel('Frequency')
    
    # Non-zero Laplacian distribution
    ax2 = plt.subplot(gs[1])
    ax2.hist(non_zero_laplacian, bins=100, color='red', alpha=0.7)
    ax2.set_title('Non-Zero Laplacian Distribution')
    ax2.set_xlabel('Laplacian Value')
    ax2.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_metrics_visualization(original, laplacian_norm, edge_binary, processing_time, kernel_size, threshold, save_path):
    """Create metrics visualization."""
    # Calculate statistics
    total_pixels = original.size
    edge_pixels = np.count_nonzero(edge_binary)
    non_edge_pixels = total_pixels - edge_pixels
    edge_density = edge_pixels / total_pixels
    
    # Calculate edge strength statistics
    edge_strength_mean = np.mean(laplacian_norm)
    edge_strength_std = np.std(laplacian_norm)
    
    # Calculate edge strength in non-zero regions only
    non_zero_laplacian = laplacian_norm[laplacian_norm > 0]
    if non_zero_laplacian.size > 0:
        non_zero_mean = np.mean(non_zero_laplacian)
        non_zero_std = np.std(non_zero_laplacian)
    else:
        non_zero_mean = 0
        non_zero_std = 0
    
    plt.figure(figsize=(12, 10))
    
    # Create a 2x2 grid of subplots
    gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    
    # Pixel distribution
    ax1 = plt.subplot(gs[0, 0])
    ax1.bar(['Edge Pixels', 'Non-Edge Pixels'], [edge_pixels, non_edge_pixels], color=['red', 'blue'])
    ax1.set_title('Pixel Distribution')
    ax1.set_ylabel('Number of Pixels')
    
    # Edge density
    ax2 = plt.subplot(gs[0, 1])
    ax2.bar(['Edge Density'], [edge_density], color='purple')
    ax2.set_title('Edge Density')
    ax2.set_ylabel('Density')
    ax2.set_ylim([0, 1])
    
    # Edge strength statistics
    ax3 = plt.subplot(gs[1, 0])
    metrics = ['Mean Edge Strength', 'Std Edge Strength', 'Non-Zero Mean', 'Non-Zero Std']
    values = [edge_strength_mean, edge_strength_std, non_zero_mean, non_zero_std]
    ax3.bar(metrics, values, color=['orange', 'cyan', 'magenta', 'yellow'])
    ax3.set_title('Edge Strength Statistics')
    ax3.set_ylabel('Value')
    
    # Processing metrics
    ax4 = plt.subplot(gs[1, 1])
    metrics = ['Kernel Size', 'Processing Time (ms)']
    values = [kernel_size, processing_time * 1000]
    ax4.bar(metrics, values, color=['green', 'brown'])
    ax4.set_title('Processing Metrics')
    ax4.set_ylabel('Value')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_kernel_analysis(original, save_path):
    """Create kernel analysis showing the effect of different kernel sizes."""
    plt.figure(figsize=(16, 10))
    
    # Create a 2x3 grid of subplots
    gs = GridSpec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1])
    
    # Original image
    ax = plt.subplot(gs[0, 0])
    ax.imshow(original, cmap='gray')
    ax.set_title('Original Image')
    ax.axis('off')
    
    # Test different kernel sizes
    kernel_sizes = [1, 3, 5, 7]
    for i, ksize in enumerate(kernel_sizes):
        row = (i + 1) // 3
        col = (i + 1) % 3
        
        if row == 0 and col == 0:
            continue  # Skip the first position as it's used for the original image
        
        ax = plt.subplot(gs[row, col])
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(original, (3, 3), 0)
        
        # Apply Laplacian with the current kernel size
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=ksize)
        laplacian_abs = np.absolute(laplacian)
        laplacian_norm = cv2.normalize(laplacian_abs, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        ax.imshow(laplacian_norm, cmap='gray')
        ax.set_title(f'Kernel Size: {ksize}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_edge_overlay(color_image, edge_binary, save_path, output_format, quality):
    """Create edge overlay visualization."""
    # Convert edge binary to 3-channel for overlay
    if len(edge_binary.shape) == 2:
        edge_color = cv2.cvtColor(edge_binary, cv2.COLOR_GRAY2BGR)
    else:
        edge_color = edge_binary.copy()
    
    # Create a red edge mask
    edge_mask = np.zeros_like(edge_color)
    edge_mask[edge_binary > 0] = [0, 0, 255]  # Red edges
    
    # Blend the original image with the edge mask
    alpha = 0.7  # Transparency factor
    beta = 1.0 - alpha
    overlay = cv2.addWeighted(color_image, alpha, edge_mask, beta, 0)
    
    # Save overlay image
    try:
        if output_format.lower() == 'jpg' or output_format.lower() == 'jpeg':
            cv2.imwrite(save_path, overlay, [cv2.IMWRITE_JPEG_QUALITY, quality])
        elif output_format.lower() == 'png':
            cv2.imwrite(save_path, overlay, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        else:
            cv2.imwrite(save_path, overlay)
    except Exception as e:
        raise RuntimeError(f"Error saving edge overlay image: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 8:
        print("Usage: python LED.py <input_image> <output_dir> [kernel_size] [threshold] [format] [quality] [--no-visualizations]")
        print("Example: python LED.py input.jpg output/ 3")
        print("Example: python LED.py input.jpg output/ 5 30 png 90")
        print("Use --no-visualizations to skip creating research visualizations")
        print("If threshold is not provided, Otsu's method will be used automatically")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_dir = sys.argv[2]
    
    # Get optional parameters
    kernel_size = 3  # Default kernel size
    threshold = None  # Default threshold (None for Otsu's method)
    output_format = 'jpg'  # Default format
    quality = 95  # Default quality
    create_visualizations = True  # Default to create visualizations
    
    # Parse kernel_size if provided
    if len(sys.argv) >= 4 and sys.argv[3] != "--no-visualizations":
        try:
            kernel_size = int(sys.argv[3])
            # Validate kernel size
            if kernel_size not in [1, 3, 5, 7]:
                print("Warning: Invalid kernel_size. Using default (3). Valid sizes are 1, 3, 5, 7.")
                kernel_size = 3
        except ValueError:
            print("Warning: Invalid kernel_size value, using default (3)")
            kernel_size = 3
    
    # Parse threshold if provided
    if len(sys.argv) >= 5 and sys.argv[4] != "--no-visualizations":
        try:
            threshold = int(sys.argv[4])
            # Validate threshold
            if threshold < 0 or threshold > 255:
                print("Warning: Invalid threshold. Using Otsu's method.")
                threshold = None
        except ValueError:
            print("Warning: Invalid threshold value, using Otsu's method")
            threshold = None
    
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
        edge_img, metadata = laplacian_edge_detection(
            input_path, output_dir, kernel_size, threshold,
            output_format, quality, create_visualizations
        )
        
        print("Laplacian edge detection complete. Metadata:")
        for key, value in metadata.items():
            print(f"{key}: {value}")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
