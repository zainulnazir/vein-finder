import cv2
import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.gridspec import GridSpec

def convert_to_grayscale(image_path, save_path, output_format='jpg', quality=95, create_visualizations=True):
    """
    Convert a color image to grayscale with comprehensive research visualizations.
    
    Args:
        image_path: Path to the input color image
        save_path: Path to save the grayscale image or directory
        output_format: Output image format ('jpg', 'png', etc.)
        quality: JPEG quality (0-100), only applicable for JPEG
        create_visualizations: Whether to create research visualizations
    
    Returns:
        tuple: (grayscale_image, metadata_dict)
    """
    # Check if input file exists
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Read the image
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
    except Exception as e:
        raise RuntimeError(f"Error reading image: {str(e)}")

    # Convert to grayscale
    start_time = time.time()
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    conversion_time = time.time() - start_time

    # Ensure save_path has a valid filename and extension
    if os.path.isdir(save_path):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(save_path, f"{base_name}_gray.{output_format}")
    else:
        # Ensure the directory exists
        output_dir = os.path.dirname(save_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

    # Save the grayscale image with appropriate parameters
    try:
        if output_format.lower() == 'jpg' or output_format.lower() == 'jpeg':
            cv2.imwrite(save_path, gray_image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        elif output_format.lower() == 'png':
            cv2.imwrite(save_path, gray_image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        else:
            cv2.imwrite(save_path, gray_image)
    except Exception as e:
        raise RuntimeError(f"Error saving image: {str(e)}")

    # Initialize visualization paths
    comparison_path = None
    histogram_path = None
    intensity_path = None
    metrics_path = None
    channel_path = None
    edge_path = None

    # Create comparison image
    comparison_path = save_path.replace(f"_gray.{output_format}", f"_comparison.{output_format}")
    create_comparison_image(image, gray_image, comparison_path, output_format, quality)

    # Create research visualizations if requested
    if create_visualizations:
        # Create histogram comparison
        histogram_path = save_path.replace(f"_gray.{output_format}", f"_histograms.png")
        create_histogram_plot(image, gray_image, histogram_path)
        
        # Create intensity distribution plot
        intensity_path = save_path.replace(f"_gray.{output_format}", f"_intensity_distribution.png")
        create_intensity_distribution(image, gray_image, intensity_path)
        
        # Create metrics visualization
        metrics_path = save_path.replace(f"_gray.{output_format}", f"_metrics.png")
        create_metrics_visualization(image, gray_image, conversion_time, metrics_path)
        
        # Create channel contribution analysis
        channel_path = save_path.replace(f"_gray.{output_format}", f"_channel_analysis.png")
        create_channel_analysis(image, gray_image, channel_path)
        
        # Create edge preservation analysis
        edge_path = save_path.replace(f"_gray.{output_format}", f"_edge_analysis.png")
        create_edge_analysis(image, gray_image, edge_path)

    # Get output file size
    output_stats = os.stat(save_path)
    
    # Calculate image statistics
    gray_mean = np.mean(gray_image)
    gray_std = np.std(gray_image)
    gray_min = np.min(gray_image)
    gray_max = np.max(gray_image)
    
    # Calculate original image statistics
    if len(image.shape) == 3:  # Color image
        b, g, r = cv2.split(image)
        orig_mean = [np.mean(b), np.mean(g), np.mean(r)]
        orig_std = [np.std(b), np.std(g), np.std(r)]
    else:  # Already grayscale
        orig_mean = [np.mean(image)]
        orig_std = [np.std(image)]
    
    # Calculate information loss (simplified metric)
    if len(image.shape) == 3:  # Color image
        # Reconstruct color image from grayscale to estimate information loss
        reconstructed = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        mse = np.mean((image - reconstructed) ** 2)
        information_loss = mse
    else:  # Already grayscale
        information_loss = 0
    
    # Gather metadata
    input_stats = os.stat(image_path)
    height, width = gray_image.shape
    info = {
        "Original Image Path": image_path,
        "Converted Image Path": save_path,
        "Comparison Image Path": comparison_path,
        "Histogram Plot Path": histogram_path,
        "Intensity Distribution Path": intensity_path,
        "Metrics Visualization Path": metrics_path,
        "Channel Analysis Path": channel_path,
        "Edge Analysis Path": edge_path,
        "Original Dimensions": image.shape[:2],
        "Grayscale Dimensions": (height, width),
        "Original File Size (KB)": round(input_stats.st_size / 1024, 2),
        "Converted File Size (KB)": round(output_stats.st_size / 1024, 2),
        "Size Reduction (%)": round((1 - output_stats.st_size / input_stats.st_size) * 100, 2),
        "Image Depth": image.dtype,
        "Conversion Time (ms)": round(conversion_time * 1000, 2),
        "Conversion Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Grayscale Mean Intensity": round(gray_mean, 2),
        "Grayscale Std Deviation": round(gray_std, 2),
        "Grayscale Min Intensity": gray_min,
        "Grayscale Max Intensity": gray_max,
        "Original Mean Intensity (B,G,R)": [round(m, 2) for m in orig_mean] if len(orig_mean) == 3 else round(orig_mean[0], 2),
        "Original Std Deviation (B,G,R)": [round(s, 2) for s in orig_std] if len(orig_std) == 3 else round(orig_std[0], 2),
        "Information Loss (MSE)": round(information_loss, 2)
    }

    return gray_image, info

def create_comparison_image(image, gray_image, save_path, output_format, quality):
    """Create a side-by-side comparison of original and grayscale images."""
    # Convert grayscale back to BGR for side-by-side comparison
    gray_bgr = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    
    # Get dimensions
    h, w = image.shape[:2]
    
    # Create a new image with twice the width
    comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
    
    # Place original image on the left
    comparison[:, :w] = image
    
    # Place grayscale image on the right
    comparison[:, w:] = gray_bgr
    
    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    text_color = (255, 255, 255)  # White
    
    # Add "Original" label
    cv2.putText(comparison, "Original", (10, 30), font, font_scale, text_color, font_thickness)
    
    # Add "Grayscale" label
    cv2.putText(comparison, "Grayscale", (w + 10, 30), font, font_scale, text_color, font_thickness)
    
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

def create_histogram_plot(image, gray_image, save_path):
    """Create histogram comparison plot."""
    plt.figure(figsize=(12, 8))
    
    # Create a 2x2 grid of subplots
    gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    
    # Original image
    ax1 = plt.subplot(gs[0, 0])
    if len(image.shape) == 3:  # Color image
        ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:  # Grayscale
        ax1.imshow(image, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Grayscale image
    ax2 = plt.subplot(gs[0, 1])
    ax2.imshow(gray_image, cmap='gray')
    ax2.set_title('Grayscale Image')
    ax2.axis('off')
    
    # Original histogram
    ax3 = plt.subplot(gs[1, 0])
    if len(image.shape) == 3:  # Color image
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            ax3.plot(hist, color=col)
            ax3.set_xlim([0, 256])
        ax3.set_title('Original Histogram (RGB)')
        ax3.set_xlabel('Pixel Value')
        ax3.set_ylabel('Frequency')
    else:  # Grayscale
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        ax3.plot(hist, color='gray')
        ax3.set_xlim([0, 256])
        ax3.set_title('Original Histogram')
        ax3.set_xlabel('Pixel Value')
        ax3.set_ylabel('Frequency')
    
    # Grayscale histogram
    ax4 = plt.subplot(gs[1, 1])
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    ax4.plot(hist, color='gray')
    ax4.set_xlim([0, 256])
    ax4.set_title('Grayscale Histogram')
    ax4.set_xlabel('Pixel Value')
    ax4.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_intensity_distribution(image, gray_image, save_path):
    """Create intensity distribution plot."""
    plt.figure(figsize=(12, 6))
    
    # Create a 1x2 grid of subplots
    gs = GridSpec(1, 2, width_ratios=[1, 1])
    
    # Original intensity distribution
    ax1 = plt.subplot(gs[0])
    if len(image.shape) == 3:  # Color image
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            channel = image[:, :, i]
            ax1.hist(channel.flatten(), bins=50, alpha=0.5, color=col, label=f'Channel {col.upper()}')
        ax1.set_title('Original Intensity Distribution')
        ax1.set_xlabel('Pixel Value')
        ax1.set_ylabel('Frequency')
        ax1.legend()
    else:  # Grayscale
        ax1.hist(image.flatten(), bins=50, color='gray', alpha=0.7)
        ax1.set_title('Original Intensity Distribution')
        ax1.set_xlabel('Pixel Value')
        ax1.set_ylabel('Frequency')
    
    # Grayscale intensity distribution
    ax2 = plt.subplot(gs[1])
    ax2.hist(gray_image.flatten(), bins=50, color='gray', alpha=0.7)
    ax2.set_title('Grayscale Intensity Distribution')
    ax2.set_xlabel('Pixel Value')
    ax2.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_metrics_visualization(image, gray_image, conversion_time, save_path):
    """Create metrics visualization."""
    # Calculate metrics
    gray_mean = np.mean(gray_image)
    gray_std = np.std(gray_image)
    gray_min = np.min(gray_image)
    gray_max = np.max(gray_image)
    
    # Calculate original image metrics
    if len(image.shape) == 3:  # Color image
        b, g, r = cv2.split(image)
        orig_mean = [np.mean(b), np.mean(g), np.mean(r)]
        orig_std = [np.std(b), np.std(g), np.std(r)]
    else:  # Already grayscale
        orig_mean = [np.mean(image)]
        orig_std = [np.std(image)]
    
    # Calculate information loss
    if len(image.shape) == 3:  # Color image
        reconstructed = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        mse = np.mean((image - reconstructed) ** 2)
    else:  # Already grayscale
        mse = 0
    
    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Mean intensity comparison
    ax1 = axs[0, 0]
    if len(orig_mean) == 3:  # Color image
        ax1.bar(['Blue', 'Green', 'Red', 'Grayscale'], 
                [orig_mean[0], orig_mean[1], orig_mean[2], gray_mean],
                color=['blue', 'green', 'red', 'gray'])
    else:  # Grayscale
        ax1.bar(['Original', 'Grayscale'], 
                [orig_mean[0], gray_mean],
                color=['gray', 'darkgray'])
    ax1.set_title('Mean Intensity Comparison')
    ax1.set_ylabel('Mean Intensity')
    
    # Standard deviation comparison
    ax2 = axs[0, 1]
    if len(orig_std) == 3:  # Color image
        ax2.bar(['Blue', 'Green', 'Red', 'Grayscale'], 
                [orig_std[0], orig_std[1], orig_std[2], gray_std],
                color=['blue', 'green', 'red', 'gray'])
    else:  # Grayscale
        ax2.bar(['Original', 'Grayscale'], 
                [orig_std[0], gray_std],
                color=['gray', 'darkgray'])
    ax2.set_title('Standard Deviation Comparison')
    ax2.set_ylabel('Standard Deviation')
    
    # Intensity range comparison
    ax3 = axs[1, 0]
    if len(image.shape) == 3:  # Color image
        orig_min = [np.min(b), np.min(g), np.min(r)]
        orig_max = [np.max(b), np.max(g), np.max(r)]
        ax3.bar(['Blue Min', 'Green Min', 'Red Min', 'Gray Min'], 
                [orig_min[0], orig_min[1], orig_min[2], gray_min],
                color=['blue', 'green', 'red', 'gray'])
        ax3.bar(['Blue Max', 'Green Max', 'Red Max', 'Gray Max'], 
                [orig_max[0], orig_max[1], orig_max[2], gray_max],
                color=['lightblue', 'lightgreen', 'lightcoral', 'lightgray'])
    else:  # Grayscale
        orig_min = np.min(image)
        orig_max = np.max(image)
        ax3.bar(['Original Min', 'Grayscale Min'], 
                [orig_min, gray_min],
                color=['gray', 'darkgray'])
        ax3.bar(['Original Max', 'Grayscale Max'], 
                [orig_max, gray_max],
                color=['lightgray', 'silver'])
    ax3.set_title('Intensity Range Comparison')
    ax3.set_ylabel('Pixel Value')
    
    # Conversion metrics
    ax4 = axs[1, 1]
    metrics = ['Conversion Time (ms)', 'Information Loss (MSE)']
    values = [conversion_time * 1000, mse]
    ax4.bar(metrics, values, color=['green', 'red'])
    ax4.set_title('Conversion Metrics')
    ax4.set_ylabel('Value')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_channel_analysis(image, gray_image, save_path):
    """Create channel contribution analysis."""
    if len(image.shape) != 3:  # Not a color image
        # Create a simple plot showing it's already grayscale
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, 'Image is already grayscale', 
                horizontalalignment='center', verticalalignment='center',
                fontsize=16)
        plt.title('Channel Analysis')
        plt.axis('off')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    # Split the image into B, G, R channels
    b, g, r = cv2.split(image)
    
    # Calculate correlation between each channel and the grayscale image
    corr_b = np.corrcoef(b.flatten(), gray_image.flatten())[0, 1]
    corr_g = np.corrcoef(g.flatten(), gray_image.flatten())[0, 1]
    corr_r = np.corrcoef(r.flatten(), gray_image.flatten())[0, 1]
    
    # Calculate standard OpenCV weights for reference
    # OpenCV uses: Y = 0.299*R + 0.587*G + 0.114*B
    opencv_weights = [0.114, 0.587, 0.299]  # B, G, R
    
    # Calculate actual contribution based on correlation
    total_corr = abs(corr_b) + abs(corr_g) + abs(corr_r)
    if total_corr > 0:
        actual_contrib = [abs(corr_b)/total_corr, abs(corr_g)/total_corr, abs(corr_r)/total_corr]
    else:
        actual_contrib = [1/3, 1/3, 1/3]
    
    plt.figure(figsize=(12, 8))
    
    # Create a 2x2 grid of subplots
    gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    
    # Channel images
    ax1 = plt.subplot(gs[0, 0])
    ax1.imshow(b, cmap='Blues')
    ax1.set_title('Blue Channel')
    ax1.axis('off')
    
    ax2 = plt.subplot(gs[0, 1])
    ax2.imshow(g, cmap='Greens')
    ax2.set_title('Green Channel')
    ax2.axis('off')
    
    # Channel comparison
    ax3 = plt.subplot(gs[1, 0])
    channels = ['Blue', 'Green', 'Red']
    x = np.arange(len(channels))
    width = 0.35
    
    ax3.bar(x - width/2, opencv_weights, width, label='OpenCV Weights', color=['blue', 'green', 'red'])
    ax3.bar(x + width/2, actual_contrib, width, label='Actual Contribution', color=['lightblue', 'lightgreen', 'lightcoral'])
    
    ax3.set_xlabel('Channel')
    ax3.set_ylabel('Weight')
    ax3.set_title('Channel Contribution Analysis')
    ax3.set_xticks(x)
    ax3.set_xticklabels(channels)
    ax3.legend()
    
    # Correlation values
    ax4 = plt.subplot(gs[1, 1])
    correlations = [corr_b, corr_g, corr_r]
    ax4.bar(channels, correlations, color=['blue', 'green', 'red'])
    ax4.set_xlabel('Channel')
    ax4.set_ylabel('Correlation')
    ax4.set_title('Channel-Grayscale Correlation')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_edge_analysis(image, gray_image, save_path):
    """Create edge preservation analysis."""
    # Apply Canny edge detection
    if len(image.shape) == 3:  # Color image
        edges_color = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 100, 200)
    else:  # Grayscale
        edges_color = cv2.Canny(image, 100, 200)
    
    edges_gray = cv2.Canny(gray_image, 100, 200)
    
    # Calculate edge preservation metric
    intersection = np.logical_and(edges_color > 0, edges_gray > 0)
    union = np.logical_or(edges_color > 0, edges_gray > 0)
    
    if np.sum(union) > 0:
        jaccard = np.sum(intersection) / np.sum(union)
    else:
        jaccard = 1.0  # Both have no edges
    
    plt.figure(figsize=(12, 8))
    
    # Create a 2x2 grid of subplots
    gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    
    # Original edges
    ax1 = plt.subplot(gs[0, 0])
    ax1.imshow(edges_color, cmap='gray')
    ax1.set_title('Original Image Edges')
    ax1.axis('off')
    
    # Grayscale edges
    ax2 = plt.subplot(gs[0, 1])
    ax2.imshow(edges_gray, cmap='gray')
    ax2.set_title('Grayscale Image Edges')
    ax2.axis('off')
    
    # Edge overlap
    ax3 = plt.subplot(gs[1, 0])
    overlap = np.zeros_like(edges_color)
    overlap[intersection] = 255  # White for overlapping edges
    overlap[edges_color > 0] = 100  # Gray for edges only in original
    overlap[edges_gray > 0] = 50  # Dark gray for edges only in grayscale
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
    if len(sys.argv) < 3 or len(sys.argv) > 6:
        print("Usage: python color2gray.py <input_image_path> <output_image_or_directory_path> [format] [quality] [--no-visualizations]")
        print("Example: python color2gray.py input.jpg output/ png")
        print("Example: python color2gray.py input.jpg output_gray.jpg jpg 90")
        print("Use --no-visualizations to skip creating research visualizations")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # Get optional parameters
    output_format = 'jpg'  # Default format
    quality = 95           # Default quality
    create_visualizations = True  # Default to create visualizations
    
    if len(sys.argv) >= 4 and sys.argv[3] != "--no-visualizations":
        output_format = sys.argv[3]
        if output_format.startswith('.'):
            output_format = output_format[1:]  # Remove dot if present
    
    if len(sys.argv) >= 5 and sys.argv[4] != "--no-visualizations":
        try:
            quality = int(sys.argv[4])
            quality = max(0, min(100, quality))  # Ensure quality is between 0-100
        except ValueError:
            print("Warning: Invalid quality value, using default (95)")
            quality = 95
    
    if "--no-visualizations" in sys.argv:
        create_visualizations = False

    try:
        gray_img, metadata = convert_to_grayscale(input_path, output_path, output_format, quality, create_visualizations)

        print("Grayscale conversion complete. Metadata:")
        for key, value in metadata.items():
            print(f"{key}: {value}")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
