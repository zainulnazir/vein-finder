import cv2
import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt
from datetime import datetime

def enhance_veins_with_gabor(image, ksize=31, sigma=4.0, lambd=10.0, gamma=0.5):
    """
    Enhance vein-like structures using a Gabor filter bank.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    filters = []
    num_filters = 16
    for theta in np.arange(0, np.pi, np.pi / num_filters):
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
        kern /= 1.5 * kern.sum()
        filters.append(kern)

    accum = np.zeros_like(gray, dtype=np.float32)
    for kern in filters:
        fimg = cv2.filter2D(gray, cv2.CV_32F, kern)
        np.maximum(accum, fimg, accum)

    accum = cv2.normalize(accum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    final_image = clahe.apply(accum)

    return final_image

def create_histogram_analysis(original, enhanced, save_path):
    """Generate and save a histogram comparison plot."""
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(original.ravel(), 256, [0, 256], color='gray', alpha=0.7)
    plt.title('Original Image Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(enhanced.ravel(), 256, [0, 256], color='blue', alpha=0.7)
    plt.title('Enhanced Image Histogram')
    plt.xlabel('Pixel Intensity')
    
    plt.suptitle('Pixel Intensity Distribution', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    plt.close()

def create_intensity_profile(original, enhanced, save_path):
    """Generate and save an intensity profile plot from the image center."""
    center_row = original.shape[0] // 2
    
    plt.figure(figsize=(12, 6))
    plt.plot(original[center_row, :], color='gray', label='Original', alpha=0.8)
    plt.plot(enhanced[center_row, :], color='blue', label='Enhanced', alpha=0.8)
    plt.title('Pixel Intensity Profile (Center Row)')
    plt.xlabel('Pixel Column')
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def calculate_metrics(image):
    """Calculate mean, std dev, and Michelson contrast."""
    mean = np.mean(image)
    std = np.std(image)
    min_val, max_val = np.min(image), np.max(image)
    contrast = (max_val - min_val) / (max_val + min_val) if (max_val + min_val) > 0 else 0
    return {
        "Mean": round(mean, 2),
        "Std Dev": round(std, 2),
        "Michelson Contrast": round(contrast, 4)
    }

def run_vein_detection(input_path, output_dir, output_format='jpg', quality=95):
    """
    Main function to run the vein detection and save outputs and research data.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Error: Could not read image at {input_path}")

    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Could not read image at {input_path}")

    start_time = time.time()
    original_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    veins_enhanced = enhance_veins_with_gabor(original_gray)
    processing_time = time.time() - start_time

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    # --- Save Images ---
    output_path = os.path.join(output_dir, f"{base_name}_veins_enhanced.{output_format}")
    cv2.imwrite(output_path, veins_enhanced)
    
    comparison_path = os.path.join(output_dir, f"{base_name}_comparison.{output_format}")
    cv2.imwrite(comparison_path, np.hstack([original_gray, veins_enhanced]))
    
    overlay_path = os.path.join(output_dir, f"{base_name}_overlay.{output_format}")
    overlay_image = cv2.cvtColor(original_gray, cv2.COLOR_GRAY2BGR)
    red_mask = np.zeros_like(overlay_image)
    red_mask[:, :, 2] = veins_enhanced
    overlay_blended = cv2.addWeighted(overlay_image, 0.6, red_mask, 0.4, 0)
    cv2.imwrite(overlay_path, overlay_blended)

    # --- Generate and Save Research Data ---
    histogram_path = os.path.join(output_dir, f"{base_name}_histogram.png")
    create_histogram_analysis(original_gray, veins_enhanced, histogram_path)

    profile_path = os.path.join(output_dir, f"{base_name}_intensity_profile.png")
    create_intensity_profile(original_gray, veins_enhanced, profile_path)

    # --- Calculate and Gather Metadata ---
    original_metrics = calculate_metrics(original_gray)
    enhanced_metrics = calculate_metrics(veins_enhanced)
    
    height, width = image.shape[:2]
    info = {
        "Input Image Path": input_path,
        "Enhanced Vein Image Path": output_path,
        "Comparison Image Path": comparison_path,
        "Overlay Image Path": overlay_path,
        "Histogram Plot Path": histogram_path,
        "Intensity Profile Plot Path": profile_path,
        "Image Dimensions": (height, width),
        "Original Image Metrics": original_metrics,
        "Enhanced Image Metrics": enhanced_metrics,
        "Processing Time (ms)": round(processing_time * 1000, 2),
        "Processing Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    return veins_enhanced, info

def experiment_with_parameters(input_path, output_dir):
    """
    Experiment with different Gabor filter parameters.
    """
    print("Starting Gabor filter parameter experimentation...")
    exp_dir = os.path.join(output_dir, "experiments")
    os.makedirs(exp_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    sigmas = [4.0, 6.0, 8.0]
    lambdas = [10.0, 15.0, 20.0]
    
    fig, axes = plt.subplots(len(sigmas), len(lambdas), figsize=(16, 12))
    fig.suptitle('Gabor Filter Experiment (ksize=31, gamma=0.5)', fontsize=20)
    
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image at {input_path}")
        
    for i, sigma in enumerate(sigmas):
        for j, lambd in enumerate(lambdas):
            veins_enhanced = enhance_veins_with_gabor(image, sigma=sigma, lambd=lambd)
            ax = axes[i, j]
            ax.imshow(veins_enhanced, cmap='gray')
            ax.set_title(f'Sigma={sigma}, Lambda={lambd}')
            ax.axis('off')
            exp_path = os.path.join(exp_dir, f"{base_name}_s{sigma}_l{lambd}.jpg")
            cv2.imwrite(exp_path, veins_enhanced)
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    grid_path = os.path.join(output_dir, f"{base_name}_parameter_grid.png")
    plt.savefig(grid_path, dpi=300)
    plt.close()
    
    print(f"Parameter experimentation complete. Individual results saved to {exp_dir}")
    print(f"Parameter grid saved to {grid_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python vein_detector.py <input_image> <output_dir> [--experiment]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2]

    if "--experiment" in sys.argv:
        try:
            experiment_with_parameters(input_path, output_dir)
        except Exception as e:
            print(f"An error occurred during experimentation: {e}")
            sys.exit(1)
    else:
        try:
            processed_img, metadata = run_vein_detection(input_path, output_dir)
            print("\nVein detection complete. Metadata:")
            for key, value in metadata.items():
                if isinstance(value, dict):
                    print(f"- {key}:")
                    for sub_key, sub_value in value.items():
                        print(f"  - {sub_key}: {sub_value}")
                else:
                    print(f"- {key}: {value}")
        except Exception as e:
            print(f"An error occurred: {e}")
            sys.exit(1)

