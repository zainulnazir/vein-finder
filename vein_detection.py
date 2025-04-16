#!/usr/bin/env python3

import cv2
import numpy as np
from skimage import (
    exposure,
    filters,
    feature,
)  # Added feature import for peak_local_max
from datetime import datetime


class VeinDetector:
    def __init__(self):
        """Initialize vein detector with default parameters"""
        # Default CLAHE object, will be recreated if settings change
        self.clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        # Default parameters (can be updated via settings)
        self.params = {
            "clahe_clip_limit": 5.0,
            "clahe_tile_grid_size": 8,
            "frangi_scale_min": 1.0,
            "frangi_scale_max": 5.0,
            "frangi_scale_step": 1.0,
            "frangi_beta": 0.5,
            "frangi_gamma": 15,
        }

    def update_params(self, new_params):
        """Update parameters from settings and recreate CLAHE if needed."""
        needs_clahe_update = False
        for key, value in new_params.items():
            if key in self.params:
                # Check if CLAHE params changed
                if key in ["clahe_clip_limit", "clahe_tile_grid_size"] and self.params[key] != value:
                     needs_clahe_update = True
                self.params[key] = value

        if needs_clahe_update:
            print(f"Recreating CLAHE: clip={self.params['clahe_clip_limit']}, grid=({self.params['clahe_tile_grid_size']},{self.params['clahe_tile_grid_size']})")
            self.clahe = cv2.createCLAHE(
                clipLimit=self.params["clahe_clip_limit"],
                tileGridSize=(self.params["clahe_tile_grid_size"], self.params["clahe_tile_grid_size"])
            )

    def preprocess(self, image):
        """Preprocess the image for vein detection"""
        # Convert to grayscale if needed
        if len(image.shape) > 2 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Apply CLAHE (using potentially updated object)
        clahe_img = self.clahe.apply(gray)

        # Light denoising (consider making this optional/configurable)
        # blurred = cv2.GaussianBlur(clahe_img, (3, 3), 0) # Keep simple for now
        # Use median blur as it might be better for salt-pepper noise from sensor/IR
        blurred = cv2.medianBlur(clahe_img, 3)

        return blurred

    def detect_veins(self, image, method="adaptive"):
        """Detect veins in the image using specified method"""
        # Preprocessing uses potentially updated CLAHE settings
        preprocessed = self.preprocess(image)

        # Pass current parameters to the methods that need them
        if method == "adaptive":
            # Adaptive method doesn't use specific tunable params from self.params currently
            return self._adaptive_threshold_method(preprocessed)
        elif method == "frangi":
            return self._frangi_filter_method(preprocessed, self.params)
        elif method == "laplacian":
            # Laplacian method doesn't use specific tunable params
            return self._laplacian_method(preprocessed)
        else:
            print(f"Unknown method '{method}', falling back to adaptive.")
            return self._adaptive_threshold_method(preprocessed) # Default

    def _adaptive_threshold_method(self, image):
        """Use adaptive thresholding to detect veins - based on working example"""
        # Create intensity mask to exclude shadows
        intensity_mask = (image > 50).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        intensity_mask = cv2.morphologyEx(
            intensity_mask, cv2.MORPH_OPEN, kernel, iterations=2
        )

        # Apply adaptive thresholding
        adaptive_thresh = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            41, # Block size - could be tunable
            15, # Constant C - could be tunable
        )

        # Refine vein mask with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        morph = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)
        vein_mask = (morph == 255).astype(np.uint8)

        # Apply intensity mask to exclude shadows and hair
        vein_mask = cv2.bitwise_and(vein_mask, intensity_mask)

        # Create a processed frame with a bright body
        background_adjusted = cv2.convertScaleAbs(image, alpha=1.0, beta=10)
        processed_frame = background_adjusted.astype(float)

        # Maximize vein darkness
        vein_darkened = cv2.convertScaleAbs(
            image, alpha=0.01, beta=-255
        )  # Nearly black veins
        processed_frame = np.where(vein_mask, vein_darkened, processed_frame)
        processed_frame = np.clip(processed_frame, 0, 255).astype(np.uint8)

        # Convert back to 3-channel for visualization
        result = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)

        return result

    def _frangi_filter_method(self, image, params):
        """Use Frangi filter with tunable parameters - NO RESIZING"""
        # Use the preprocessed image directly
        orig_img = image.copy()
        orig_height, orig_width = orig_img.shape[:2]

        # Denoising (Apply to original size image)
        denoised = cv2.medianBlur(orig_img, 5) # Denoise preprocessed image

        # Normalize image to float for the Frangi filter
        img_float = denoised.astype(float) / 255.0

        # Apply Frangi filter using parameters from settings
        scale_min = params['frangi_scale_min']
        scale_max = params['frangi_scale_max']
        scale_step = params['frangi_scale_step']
        beta = params['frangi_beta']
        gamma = params['frangi_gamma']

        # Ensure scale_max >= scale_min
        if scale_max < scale_min:
            scale_max = scale_min
        # Ensure scale_step > 0
        if scale_step <= 0:
             scale_step = 1.0

        scales = np.arange(scale_min, scale_max + scale_step, scale_step)
        if len(scales) == 0:
             scales = np.array([scale_min])

        print(f"Frangi Scales: {scales}, Beta: {beta}, Gamma: {gamma}")

        vein_enhanced = filters.frangi(
            img_float,
            sigmas=scales,
            beta=beta,
            gamma=gamma,
            black_ridges=True,
        )

        # Normalize Frangi output to 0-1 range
        if np.max(vein_enhanced) > 0:
             vein_enhanced = vein_enhanced / np.max(vein_enhanced)

        # Invert (bright veins) and scale to 0-255 uint8
        vein_enhanced = 1.0 - vein_enhanced
        vein_enhanced = exposure.rescale_intensity(vein_enhanced, out_range=(0, 255)).astype(np.uint8)

        # Apply CLAHE *after* Frangi to enhance the result contrast
        final_frangi_output = self.clahe.apply(vein_enhanced) # Apply CLAHE here
        print("Applied CLAHE to final Frangi output.")

        # Return as 3-channel grayscale image
        result = cv2.cvtColor(final_frangi_output, cv2.COLOR_GRAY2BGR)
        return result

    def _laplacian_method(self, image):
        """Use Laplacian filter with blended overlay."""
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))

        # Increase threshold to reduce noise sensitivity
        threshold_value = 25 # Increased from 10
        _, binary = cv2.threshold(laplacian, threshold_value, 255, cv2.THRESH_BINARY)

        # Use a slightly larger kernel for closing/opening
        close_kernel = np.ones((3, 3), np.uint8) # Keep 3x3 for closing gaps
        open_kernel = np.ones((5, 5), np.uint8) # Use 5x5 for opening to remove noise

        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_kernel)

        # --- Visualization (Blended Overlay) ---
        background = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # Create blue mask where veins are detected
        mask = np.zeros_like(background)
        mask[cleaned == 255] = [255, 0, 0] # Blue color
        # Blend the mask with the background
        result = cv2.addWeighted(background, 0.8, mask, 0.5, 0) # Adjust weights as needed

        return result

    def enhance_contrast(self, image, method="clahe", clip_limit=3.0):
        """DEPRECATED? Contrast enhancement is now part of preprocess"""
        if method == "clahe":
            # Use the class's CLAHE object
            return self.clahe.apply(image)
        elif method == "histogram_equalization":
            return cv2.equalizeHist(image)
        else:
            return image

# Example usage (if run directly)
if __name__ == '__main__':
    # Load a sample image
    sample_image_path = 'static/images/sample_vein_image.jpg' # Replace with a real path
    if not os.path.exists(sample_image_path):
         print(f"Sample image not found at {sample_image_path}, using placeholder.")
         sample_image = np.random.randint(50, 150, size=(480, 640), dtype=np.uint8)
    else:
         sample_image = cv2.imread(sample_image_path, cv2.IMREAD_GRAYSCALE)

    detector = VeinDetector()

    # Update parameters if needed
    # detector.update_params({"clahe_clip_limit": 4.0})

    methods = ["adaptive", "frangi", "laplacian"]
    for method_name in methods:
        print(f"--- Processing with {method_name} ---")
        processed = detector.detect_veins(sample_image, method=method_name)

        # Display or save the result
        cv2.imshow(f"Original", sample_image)
        cv2.imshow(f"Processed - {method_name}", processed)
        key = cv2.waitKey(0) # Wait indefinitely for a key press
        if key == 27: # Check if ESC key was pressed
            break
        cv2.destroyWindow(f"Processed - {method_name}") # Close current window

    cv2.destroyAllWindows()
