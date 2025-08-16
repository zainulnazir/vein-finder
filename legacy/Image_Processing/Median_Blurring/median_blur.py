import cv2
import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt
from datetime import datetime

def apply_median_blur(image_path, save_path, ksize=5, output_format='jpg', quality=95, create_visualizations=True):
    """
    Apply median blurring to an image with research visualizations focused on noise reduction.
    
    Args:
        image_path: Path to the input image
        save_path: Path to save the processed image or directory
        ksize: Kernel size for median blur (must be odd and greater than 1)
        output_format: Output image format ('jpg', 'png', etc.)
        quality: JPEG quality (0-100), only applicable for JPEG
        create_visualizations: Whether to create research visualizations
    
    Returns:
        tuple: (blurred_image, metadata_dict)
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

    # Ensure kernel size is odd and greater than 1
    if ksize % 2 == 0:
        ksize += 1
    if ksize <= 1:
        ksize = 3

    # Apply median blur
    start_time = time.time()
    blurred_image = cv2.medianBlur(image, ksize)
    processing_time = time.time() - start_time

    # Ensure save_path has a valid filename and extension
    if os.path.isdir(save_path):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(save_path, f"{base_name}_medianblur.{output_format}")
    else:
        # Ensure the directory exists
        output_dir = os.path.dirname(save_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

    # Save the processed image with appropriate parameters
    try:
        if output_format.lower() == 'jpg' or output_format.lower() == 'jpeg':
            cv2.imwrite(save_path, blurred_image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        elif output_format.lower() == 'png':
            cv2.imwrite(save_path, blurred_image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        else:
            cv2.imwrite(save_path, blurred_image)
    except Exception as e:
        raise RuntimeError(f"Error saving image: {str(e)}")

    # Initialize visualization paths
    comparison_path = None
    noise_analysis_path = None
    metrics_path = None

    # Create comparison image
    comparison_path = save_path.replace(f"_medianblur.{output_format}", f"_comparison.{output_format}")
    create_comparison_image(image, blurred_image, comparison_path, output_format, quality)

    # Create research visualizations if requested
    if create_visualizations:
        # Create noise analysis visualization
        noise_analysis_path = save_path.replace(f"_medianblur.{output_format}", f"_noise_analysis.png")
        create_noise_analysis(image, blurred_image, ksize, noise_analysis_path)
        
        # Create metrics visualization
        metrics_path = save_path.replace(f"_medianblur.{output_format}", f"_metrics.png")
        create_metrics_visualization(image, blurred_image, processing_time, ksize, metrics_path)

    # Get output file size
    output_stats = os.stat(save_path)
    
    # Calculate image statistics
    if len(image.shape) == 3:  # Color image
        orig_b, orig_g, orig_r = cv2.split(image)
        blur_b, blur_g, blur_r = cv2.split(blurred_image)
        
        orig_mean = [np.mean(orig_b), np.mean(orig_g), np.mean(orig_r)]
        orig_std = [np.std(orig_b), np.std(orig_g), np.std(orig_r)]
        blur_mean = [np.mean(blur_b), np.mean(blur_g), np.mean(blur_r)]
        blur_std = [np.std(blur_b), np.std(blur_g), np.std(blur_r)]
    else:  # Grayscale
        orig_mean = [np.mean(image)]
        orig_std = [np.std(image)]
        blur_mean = [np.mean(blurred_image)]
        blur_std = [np.std(blurred_image)]
    
    # Calculate noise reduction
    noise_reduction = []
    for i in range(len(orig_std)):
        reduction = ((orig_std[i] - blur_std[i]) / orig_std[i]) * 100 if orig_std[i] > 0 else 0
        noise_reduction.append(reduction)
    
    # Gather metadata
    input_stats = os.stat(image_path)
    height, width = blurred_image.shape[:2]
    info = {
        "Original Image Path": image_path,
        "Processed Image Path": save_path,
        "Comparison Image Path": comparison_path,
        "Noise Analysis Path": noise_analysis_path,
        "Metrics Visualization Path": metrics_path,
        "Image Dimensions": (height, width),
        "Original File Size (KB)": round(input_stats.st_size / 1024, 2),
        "Processed File Size (KB)": round(output_stats.st_size / 1024, 2),
        "File Size Change (%)": round((output_stats.st_size / input_stats.st_size - 1) * 100, 2),
        "Image Depth": image.dtype,
        "Kernel Size": ksize,
        "Processing Time (ms)": round(processing_time * 1000, 2),
        "Processing Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Original Mean Intensity (B,G,R)": [round(m, 2) for m in orig_mean] if len(orig_mean) == 3 else round(orig_mean[0], 2),
        "Original Std Deviation (B,G,R)": [round(s, 2) for s in orig_std] if len(orig_std) == 3 else round(orig_std[0], 2),
        "Blurred Mean Intensity (B,G,R)": [round(m, 2) for m in blur_mean] if len(blur_mean) == 3 else round(blur_mean[0], 2),
        "Blurred Std Deviation (B,G,R)": [round(s, 2) for s in blur_std] if len(blur_std) == 3 else round(blur_std[0], 2),
        "Noise Reduction (%) (B,G,R)": [round(r, 2) for r in noise_reduction] if len(noise_reduction) == 3 else round(noise_reduction[0], 2)
    }

    return blurred_image, info

def create_comparison_image(image, blurred_image, save_path, output_format, quality):
    """Create a side-by-side comparison of original and blurred images."""
    # Get dimensions
    h, w = image.shape[:2]
    
    # Create a new image with twice the width
    comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
    
    # Place original image on the left
    comparison[:, :w] = image
    
    # Place blurred image on the right
    comparison[:, w:] = blurred_image
    
    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    text_color = (255, 255, 255)  # White
    
    # Add "Original" label
    cv2.putText(comparison, "Original", (10, 30), font, font_scale, text_color, font_thickness)
    
    # Add "Median Blur" label
    cv2.putText(comparison, "Median Blur", (w + 10, 30), font, font_scale, text_color, font_thickness)
    
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

def create_noise_analysis(image, blurred_image, ksize, save_path):
    """Create noise analysis visualization showing the difference between original and blurred images."""
    # Calculate difference image
    if len(image.shape) == 3:  # Color image
        diff = cv2.absdiff(image, blurred_image)
        # Convert to grayscale for visualization
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    else:  # Grayscale
        diff = cv2.absdiff(image, blurred_image)
        diff_gray = diff
    
    plt.figure(figsize=(12, 8))
    
    # Create a 2x2 grid of subplots
    plt.subplot(2, 2, 1)
    if len(image.shape) == 3:  # Color image
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:  # Grayscale
        plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    if len(blurred_image.shape) == 3:  # Color image
        plt.imshow(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))
    else:  # Grayscale
        plt.imshow(blurred_image, cmap='gray')
    plt.title(f'Median Blur (ksize={ksize})')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(diff_gray, cmap='hot')
    plt.title('Difference Image (Noise)')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.hist(diff_gray.flatten(), bins=50, color='red', alpha=0.7)
    plt.title('Noise Distribution')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_metrics_visualization(image, blurred_image, processing_time, ksize, save_path):
    """Create metrics visualization focused on noise reduction."""
    # Calculate metrics
    if len(image.shape) == 3:  # Color image
        orig_b, orig_g, orig_r = cv2.split(image)
        blur_b, blur_g, blur_r = cv2.split(blurred_image)
        
        orig_std = [np.std(orig_b), np.std(orig_g), np.std(orig_r)]
        blur_std = [np.std(blur_b), np.std(blur_g), np.std(blur_r)]
    else:  # Grayscale
        orig_std = [np.std(image)]
        blur_std = [np.std(blurred_image)]
    
    # Calculate noise reduction
    noise_reduction = []
    for i in range(len(orig_std)):
        reduction = ((orig_std[i] - blur_std[i]) / orig_std[i]) * 100 if orig_std[i] > 0 else 0
        noise_reduction.append(reduction)
    
    plt.figure(figsize=(12, 6))
    
    # Create a 1x2 grid of subplots
    plt.subplot(1, 2, 1)
    metrics = ['Kernel Size', 'Processing Time (ms)']
    values = [ksize, processing_time * 1000]
    plt.bar(metrics, values, color=['blue', 'green'])
    plt.title('Processing Metrics')
    plt.ylabel('Value')
    
    plt.subplot(1, 2, 2)
    if len(orig_std) == 3:  # Color image
        plt.bar(['Blue', 'Green', 'Red'], noise_reduction, color=['blue', 'green', 'red'])
    else:  # Grayscale
        plt.bar(['Grayscale'], noise_reduction, color=['gray'])
    plt.title('Noise Reduction (%)')
    plt.ylabel('Reduction (%)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 7:
        print("Usage: python median_blur.py <input_image_path> <output_image_or_directory_path> [kernel_size] [format] [quality] [--no-visualizations]")
        print("Example: python median_blur.py input.jpg output/ 5")
        print("Example: python median_blur.py input.jpg output_blur.jpg 7 jpg 90")
        print("Use --no-visualizations to skip creating research visualizations")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # Get optional parameters
    kernel_size = 5  # Default kernel size
    output_format = 'jpg'  # Default format
    quality = 95  # Default quality
    create_visualizations = True  # Default to create visualizations
    
    # Parse kernel_size if provided
    if len(sys.argv) >= 4 and sys.argv[3] != "--no-visualizations":
        try:
            kernel_size = int(sys.argv[3])
        except ValueError:
            print("Warning: Invalid kernel_size value, using default (5)")
            kernel_size = 5
    
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
        blurred_img, metadata = apply_median_blur(
            input_path, output_path, kernel_size, 
            output_format, quality, create_visualizations
        )

        print("Median blurring complete. Metadata:")
        for key, value in metadata.items():
            print(f"{key}: {value}")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
