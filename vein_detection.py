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
        self.clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))

    def preprocess(self, image):
        """Preprocess the image for vein detection"""
        # Convert to grayscale if needed
        if len(image.shape) > 2 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe_img = self.clahe.apply(gray)

        # Light denoising
        blurred = cv2.GaussianBlur(clahe_img, (3, 3), 0)

        return blurred

    def detect_veins(self, image, method="adaptive"):
        """Detect veins in the image using specified method"""
        preprocessed = self.preprocess(image)

        if method == "adaptive":
            return self._adaptive_threshold_method(preprocessed)
        elif method == "frangi":
            return self._frangi_filter_method(preprocessed)
        elif method == "laplacian":
            return self._laplacian_method(preprocessed)
        elif method == "thermal":
            return self._thermal_profile_method(preprocessed)
        else:
            # Default to adaptive method
            return self._adaptive_threshold_method(preprocessed)

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
            41,
            15,  # Larger block size to avoid hair
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

    def _frangi_filter_method(self, image):
        """Use Frangi filter to enhance tubular structures like veins"""
        # Create a working copy of the image
        orig_img = image.copy()

        # Store original dimensions for aspect ratio preservation
        orig_height, orig_width = orig_img.shape[:2]
        orig_aspect_ratio = orig_width / orig_height

        # Performance optimization: Resize image to smaller size for processing
        # This significantly reduces computation time while maintaining feature detection
        scale_factor = 0.5
        small_width = int(orig_width * scale_factor)
        small_height = int(orig_height * scale_factor)

        # Ensure aspect ratio is maintained during resizing
        small_img = cv2.resize(
            orig_img,
            (small_width, small_height),
            interpolation=cv2.INTER_AREA,
        )

        # Apply strong contrast enhancement optimized for vein visibility
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(small_img)

        # Use faster median blur instead of bilateral filter for performance
        # Median blur still preserves edges well but is much faster
        denoised = cv2.medianBlur(enhanced, 5)

        # Normalize image to float for the Frangi filter
        img_float = denoised.astype(float) / 255.0

        # Apply Frangi filter with optimized parameters for vein detection
        # Use better parameters that work for most skin types
        vein_enhanced = filters.frangi(
            img_float,
            scale_range=(0.5, 6),  # Narrower scale range for better performance
            scale_step=2,  # Larger step size for better performance
            beta=0.9,  # Higher beta to suppress blob-like structures
            gamma=0.25,  # Lower gamma to enhance lower contrast veins
            black_ridges=True,  # Veins appear as dark structures
        )

        # Normalize back to 0-255 range
        vein_enhanced = exposure.rescale_intensity(vein_enhanced, out_range=(0, 1))
        vein_enhanced = (vein_enhanced * 255).astype(np.uint8)

        # Resize back to original size while preserving aspect ratio
        vein_enhanced = cv2.resize(
            vein_enhanced,
            (orig_width, orig_height),
            interpolation=cv2.INTER_LINEAR,
        )

        # Apply adaptive threshold to get better binary vein map
        vein_binary = cv2.adaptiveThreshold(
            vein_enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            17,  # Smaller block size for finer details
            -2,  # Negative constant for better vein detection
        )

        # Clean up the binary mask with morphological operations
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(vein_binary, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned = cv2.morphologyEx(
            cleaned, cv2.MORPH_CLOSE, kernel, iterations=1
        )  # Reduced iterations for speed

        # Create a colored output with enhanced contrast
        # Create natural skin tone background
        skin_bg = cv2.convertScaleAbs(orig_img, alpha=1.2, beta=15)
        result = cv2.cvtColor(skin_bg, cv2.COLOR_GRAY2BGR)

        # Adjust color tone for more natural skin appearance
        result = cv2.addWeighted(
            result,
            1.0,
            np.full_like(result, [30, 20, 10]),
            0.1,  # Warm skin tone adjustment
            0,
        )

        # Create a blue-ish base layer for veins
        vein_layer = np.zeros_like(result)
        vein_layer[:, :, 0] = cleaned  # Blue channel for veins

        # Apply blue vein overlay
        result = cv2.addWeighted(result, 0.9, vein_layer, 0.8, 0)

        # Add a red overlay to highlight vein centers for better visibility
        vein_centers = cv2.erode(cleaned, np.ones((3, 3), np.uint8), iterations=1)
        vein_red_overlay = np.zeros_like(result)
        vein_red_overlay[:, :, 2] = vein_centers  # Red channel

        # Blend red vein centers
        result = cv2.addWeighted(result, 0.8, vein_red_overlay, 0.6, 0)

        return result

    def _laplacian_method(self, image):
        """Use Laplacian filter to detect edges like veins"""
        # Apply Laplacian filter to detect edges
        laplacian = cv2.Laplacian(image, cv2.CV_64F)

        # Convert back to uint8
        laplacian = np.uint8(np.absolute(laplacian))

        # Threshold to get binary image
        _, binary = cv2.threshold(laplacian, 10, 255, cv2.THRESH_BINARY)

        # Clean up using morphological operations
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Create a blue mask for veins
        background = cv2.convertScaleAbs(
            image, alpha=1.0, beta=10
        )  # Brighten background
        result = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)

        # Create a blue mask for veins
        mask = np.zeros_like(result)
        mask[:, :, 0] = cleaned  # Blue channel

        # Blend with original
        alpha = 0.7
        cv2.addWeighted(result, 1, mask, alpha, 0, result)

        return result

    def _thermal_profile_method(self, image):
        """Create a realistic thermal/heat-map visualization of vein structures"""
        # Start with contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)

        # Apply bilateral filter to preserve edges while reducing noise
        filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)

        # Use adaptive thresholding with better parameters for vein detection
        vein_mask = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 23, 7
        )

        # Clean up noise with more aggressive morphological operations
        kernel = np.ones((3, 3), np.uint8)
        vein_mask = cv2.morphologyEx(vein_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        vein_mask = cv2.morphologyEx(vein_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Create better distance transform for 3D effect
        dist_transform = cv2.distanceTransform(vein_mask, cv2.DIST_L2, 5)
        dist_transform = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)

        # Apply gaussian blur for more natural heat diffusion effect
        dist_blur = cv2.GaussianBlur(dist_transform, (15, 15), 0)

        # Create a proper thermal colormap - this is the key change
        thermal_map = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

        # FLIR-style professional thermal coloring
        # Convert the image to a proper heat map that follows professional thermal camera color schemes

        # Background (cool/blue)
        blue_mask = ((1.0 - dist_blur) > 0.7).astype(np.uint8)
        blue_intensity = np.ones_like(blue_mask) * 180
        thermal_map[:, :, 0] = cv2.multiply(blue_intensity, blue_mask)  # Strong Blue

        # Cool areas (blue to purple)
        cool_mask = (((1.0 - dist_blur) <= 0.7) & ((1.0 - dist_blur) > 0.5)).astype(
            np.uint8
        )
        cool_blue = np.ones_like(cool_mask) * 120
        cool_red = np.ones_like(cool_mask) * 20
        thermal_map[:, :, 0] += cv2.multiply(cool_blue, cool_mask)  # Medium Blue
        thermal_map[:, :, 2] += cv2.multiply(cool_red, cool_mask)  # Slight Red

        # Medium temperature (green to yellow)
        medium_mask = (((1.0 - dist_blur) <= 0.5) & ((1.0 - dist_blur) > 0.3)).astype(
            np.uint8
        )
        medium_green = np.ones_like(medium_mask) * 180
        thermal_map[:, :, 1] += cv2.multiply(medium_green, medium_mask)  # Strong Green

        # Warm areas (yellow to orange)
        warm_mask = (((1.0 - dist_blur) <= 0.3) & ((1.0 - dist_blur) > 0.1)).astype(
            np.uint8
        )
        warm_red = np.ones_like(warm_mask) * 180
        warm_green = np.ones_like(warm_mask) * 120
        thermal_map[:, :, 2] += cv2.multiply(warm_red, warm_mask)  # Strong Red
        thermal_map[:, :, 1] += cv2.multiply(warm_green, warm_mask)  # Medium Green

        # Hot areas (bright red/white) - Vein centers
        hot_mask = ((1.0 - dist_blur) <= 0.1).astype(np.uint8)
        hot_red = np.ones_like(hot_mask) * 255
        hot_white = np.ones_like(hot_mask) * 70  # Add some white glow
        thermal_map[:, :, 2] += cv2.multiply(hot_red, hot_mask)  # Pure Red
        thermal_map[:, :, 1] += cv2.multiply(hot_white, hot_mask)  # Slight Green
        thermal_map[:, :, 0] += cv2.multiply(hot_white, hot_mask)  # Slight Blue

        # Edge detection for vein boundaries (makes them pop)
        edges = cv2.Canny(vein_mask, 50, 150)
        edge_intensity = np.ones_like(edges) * 255
        edge_intensity = cv2.dilate(edge_intensity, kernel, iterations=1)
        thermal_map[:, :, 2] = np.maximum(
            thermal_map[:, :, 2], edges
        )  # Add edges to red channel

        # Enhance vein centers to show as hotspots
        try:
            # Try to use ximgproc for better thinning if available
            skeleton = cv2.ximgproc.thinning(vein_mask)
        except (AttributeError, cv2.error):
            # Fallback to distance transform peak extraction
            local_max = feature.peak_local_max(
                dist_transform, min_distance=7, indices=False
            )
            skeleton = local_max.astype(np.uint8) * 255

        # Make vein centers bright white-red (hottest)
        hotspots = cv2.dilate(skeleton, np.ones((3, 3), np.uint8), iterations=1)
        thermal_map[:, :, 2] = np.maximum(thermal_map[:, :, 2], hotspots)  # Bright red
        thermal_map[:, :, 1] = np.maximum(
            thermal_map[:, :, 1], hotspots // 3
        )  # Some green
        thermal_map[:, :, 0] = np.maximum(
            thermal_map[:, :, 0], hotspots // 3
        )  # Some blue

        # Apply color blending for smoother transitions
        for i in range(3):
            thermal_map[:, :, i] = cv2.GaussianBlur(thermal_map[:, :, i], (5, 5), 0)

        # Add subtle thermal camera noise for realism
        noise = np.random.normal(0, 2, thermal_map.shape).astype(np.int8)
        thermal_map = np.clip(thermal_map + noise, 0, 255).astype(np.uint8)

        # Apply final adjustments to match FLIR camera appearance
        thermal_map = cv2.convertScaleAbs(thermal_map, alpha=1.1, beta=0)

        # Create temperature scale bar
        bar_width = 20
        bar_height = image.shape[0]
        temp_bar = np.zeros((bar_height, bar_width, 3), dtype=np.uint8)

        # Create modern thermal color scale: blue->purple->green->yellow->orange->red->white
        for y in range(bar_height):
            # Invert y coordinate (0 at top, max at bottom)
            ratio = 1.0 - (y / bar_height)

            if ratio < 0.15:  # Deep blue (coldest)
                temp_bar[y, :] = [180, 0, 0]
            elif ratio < 0.25:  # Blue to purple
                blend = (ratio - 0.15) / 0.1
                temp_bar[y, :] = [180, 0, int(100 * blend)]
            elif ratio < 0.4:  # Purple to green
                blend = (ratio - 0.25) / 0.15
                temp_bar[y, :] = [
                    int(180 * (1 - blend)),
                    int(180 * blend),
                    int(100 * (1 - blend)),
                ]
            elif ratio < 0.55:  # Green to yellow
                blend = (ratio - 0.4) / 0.15
                temp_bar[y, :] = [0, 180, int(180 * blend)]
            elif ratio < 0.7:  # Yellow to orange
                blend = (ratio - 0.55) / 0.15
                temp_bar[y, :] = [0, int(180 * (1 - blend)), 180]
            elif ratio < 0.85:  # Orange to red
                blend = (ratio - 0.7) / 0.15
                temp_bar[y, :] = [int(70 * blend), 0, 180]
            else:  # Red to white (hottest)
                blend = (ratio - 0.85) / 0.15
                temp_bar[y, :] = [70, int(70 * blend), 255]

        # Add the temperature bar
        result = np.hstack((thermal_map, temp_bar))

        # Make the temperature markers more professional looking
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        text_offset = thermal_map.shape[1] + 5

        # Add temperature scale labels with clearer visibility
        cv2.rectangle(
            result, (text_offset - 3, 5), (text_offset + 15, 25), (0, 0, 0), -1
        )
        cv2.putText(
            result,
            "°C",
            (text_offset, 15),
            font,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Add temperature values with background for better visibility
        for pos, temp in [
            (bar_height - 15, "32.0"),
            (int(bar_height * 0.8), "33.5"),
            (int(bar_height * 0.6), "35.0"),
            (int(bar_height * 0.4), "36.5"),
            (int(bar_height * 0.2), "38.0"),
        ]:
            cv2.rectangle(
                result,
                (text_offset - 3, pos - 10),
                (text_offset + 30, pos + 5),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                result,
                temp,
                (text_offset, pos),
                font,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Add professional medical thermal imaging overlay
        cv2.rectangle(result, (5, 5), (100, 50), (0, 0, 0), -1)
        cv2.putText(
            result, "THERMAL", (10, 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )
        cv2.putText(
            result, "VEIN SCAN", (10, 40), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )

        # Add timestamp for professional medical look
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.rectangle(
            result,
            (10, result.shape[0] - 25),
            (150, result.shape[0] - 5),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            result,
            timestamp,
            (15, result.shape[0] - 10),
            font,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Add "FLIR"-style logo for professional appearance
        cv2.rectangle(
            result,
            (result.shape[1] - 60, result.shape[0] - 25),
            (result.shape[1] - 10, result.shape[0] - 5),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            result,
            "VEIN-IR",
            (result.shape[1] - 55, result.shape[0] - 10),
            font,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return result

    def enhance_contrast(self, image, method="clahe", clip_limit=3.0):
        """Enhance contrast to make veins more visible"""
        # Convert to grayscale if needed
        if len(image.shape) > 2 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        if method == "clahe":
            # Create CLAHE object with specified clip limit
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
        elif method == "histogram_equalization":
            enhanced = cv2.equalizeHist(gray)
        else:
            enhanced = gray  # Default to original

        return enhanced
