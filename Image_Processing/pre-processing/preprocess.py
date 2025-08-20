import cv2
import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.gridspec import GridSpec

def preprocess_image_pipeline(image_path, output_dir, bilateral_d=9, bilateral_sigma_color=75, bilateral_sigma_space=75, clahe_clip_limit=2.0, clahe_tile_size=(8, 8), output_format='jpg', quality=95, create_visualizations=True):
    """
    A unified pipeline for preprocessing vein images, including grayscale conversion,
    bilateral filtering, and CLAHE, with comprehensive research data generation.
    """
    # --- 1. Load and Convert to Grayscale ---
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")

    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # --- 2. Apply Bilateral Filter for Noise Reduction ---
    start_time = time.time()
    filtered_image = cv2.bilateralFilter(gray_image, bilateral_d, bilateral_sigma_color, bilateral_sigma_space)
    filtering_time = time.time() - start_time

    # --- 3. Apply CLAHE for Contrast Enhancement ---
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_size)
    final_image = clahe.apply(filtered_image)
    clahe_time = time.time() - start_time - filtering_time
    total_time = time.time() - start_time

    # --- 4. Save Final Output ---
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    final_save_path = os.path.join(output_dir, f"{base_name}_preprocessed.{output_format}")
    # CORRECTED: Ensure the final, fully processed image is saved.
    cv2.imwrite(final_save_path, final_image, [cv2.IMWRITE_JPEG_QUALITY, quality])

    # --- 5. Generate Research Data and Visualizations ---
    comparison_path, noise_analysis_path, histogram_path = "N/A", "N/A", "N/A"
    if create_visualizations:
        comparison_path = os.path.join(output_dir, f"{base_name}_stages_comparison.jpg")
        stages_img = np.hstack([gray_image, filtered_image, final_image])
        cv2.imwrite(comparison_path, stages_img)

        noise_analysis_path = os.path.join(output_dir, f"{base_name}_noise_analysis.png")
        create_noise_analysis_plot(gray_image, filtered_image, noise_analysis_path)

        histogram_path = os.path.join(output_dir, f"{base_name}_histogram_analysis.png")
        create_histogram_plot(filtered_image, final_image, histogram_path)

    # --- 6. Compile Metadata ---
    gray_metrics = {'mean': np.mean(gray_image), 'std_dev': np.std(gray_image)}
    filtered_metrics = {'mean': np.mean(filtered_image), 'std_dev': np.std(filtered_image)}
    final_metrics = {'mean': np.mean(final_image), 'std_dev': np.std(final_image)}

    info = {
        "Input Image Path": image_path,
        "Final Output Path": final_save_path,
        "Parameters": {
            "Bilateral Filter": f"d={bilateral_d}, sigmaColor={bilateral_sigma_color}, sigmaSpace={bilateral_sigma_space}",
            "CLAHE": f"clipLimit={clahe_clip_limit}, tileSize={clahe_tile_size}"
        },
        "Image Statistics": {
            "Original Grayscale": f"Mean={gray_metrics['mean']:.2f}, StdDev={gray_metrics['std_dev']:.2f}",
            "After Filtering": f"Mean={filtered_metrics['mean']:.2f}, StdDev={filtered_metrics['std_dev']:.2f}",
            "After CLAHE (Final)": f"Mean={final_metrics['mean']:.2f}, StdDev={final_metrics['std_dev']:.2f}",
        },
        "Visualization Paths": {
            "Stages Comparison": comparison_path,
            "Noise Analysis": noise_analysis_path,
            "Histogram Analysis": histogram_path
        }
    }
    return final_image, info

def experiment_with_parameters(input_path, output_dir):
    """
    Experiment with different Bilateral Filter and CLAHE parameters.
    """
    print("Starting preprocessing parameter experimentation...")
    exp_dir = os.path.join(output_dir, "experiments")
    os.makedirs(exp_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    # Define parameter ranges to test
    sigma_colors = [50, 75, 100]
    clip_limits = [2.0, 3.0, 4.0]

    fig, axes = plt.subplots(len(sigma_colors), len(clip_limits), figsize=(15, 12))
    fig.suptitle('Preprocessing Experiment (Bilateral Filter + CLAHE)', fontsize=20)

    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image at {input_path}")

    for i, sigma in enumerate(sigma_colors):
        # Apply bilateral filter once per row
        filtered_image = cv2.bilateralFilter(image, 9, sigma, 75)
        for j, clip in enumerate(clip_limits):
            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
            final_image = clahe.apply(filtered_image)
            
            ax = axes[i, j]
            ax.imshow(final_image, cmap='gray')
            ax.set_title(f'Sigma={sigma}, Clip={clip}', fontsize=10)
            ax.axis('off')
            
            exp_filename = f"{base_name}_s{sigma}_c{clip}.jpg"
            exp_path = os.path.join(exp_dir, exp_filename)
            cv2.imwrite(exp_path, final_image)
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    grid_path = os.path.join(output_dir, f"{base_name}_parameter_grid.png")
    plt.savefig(grid_path, dpi=300)
    plt.close()
    
    print(f"Parameter experimentation complete. Results saved to {exp_dir}")
    print(f"Parameter grid saved to {grid_path}")

def create_noise_analysis_plot(original, filtered, save_path):
    """Creates a plot visualizing the effect of the noise filter."""
    diff_image = cv2.absdiff(original, filtered)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Grayscale')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(filtered, cmap='gray')
    plt.title('After Bilateral Filter')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(diff_image, cmap='hot')
    plt.title('Removed Noise (Difference)')
    plt.axis('off')
    plt.suptitle('Noise Reduction Analysis', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    plt.close()

def create_histogram_plot(before_clahe, after_clahe, save_path):
    """Creates a plot visualizing the effect of CLAHE."""
    plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2)
    ax1 = plt.subplot(gs[0, 0])
    ax1.imshow(before_clahe, cmap='gray')
    ax1.set_title('Before CLAHE')
    ax1.axis('off')
    ax2 = plt.subplot(gs[0, 1])
    ax2.imshow(after_clahe, cmap='gray')
    ax2.set_title('After CLAHE (Final Image)')
    ax2.axis('off')
    ax3 = plt.subplot(gs[1, :])
    ax3.hist(before_clahe.ravel(), 256, [0, 256], color='gray', alpha=0.7, label='Before CLAHE')
    ax3.hist(after_clahe.ravel(), 256, [0, 256], color='blue', alpha=0.7, label='After CLAHE')
    ax3.set_title('Histogram Comparison')
    ax3.set_xlabel('Pixel Intensity')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    plt.suptitle('Contrast Enhancement (CLAHE) Analysis', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python preprocess.py <input_image_path> <output_directory> [--experiment] [--no-visualizations]")
        print("Example: python preprocess.py my_image.jpg processed_images/")
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
        create_visualizations = True
        if "--no-visualizations" in sys.argv:
            create_visualizations = False
        try:
            final_img, metadata = preprocess_image_pipeline(
                input_path, 
                output_dir, 
                create_visualizations=create_visualizations
            )
            print("\nPreprocessing complete. Metadata:")
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

