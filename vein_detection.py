#!/usr/bin/env python3

import cv2
import numpy as np
from skimage import exposure, filters

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
        else:
            # Default to adaptive method
            return self._adaptive_threshold_method(preprocessed)
    
    def _adaptive_threshold_method(self, image):
        """Use adaptive thresholding to detect veins - based on working example"""
        # Create intensity mask to exclude shadows
        intensity_mask = (image > 50).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        intensity_mask = cv2.morphologyEx(intensity_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Apply adaptive thresholding
        adaptive_thresh = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 41, 15  # Larger block size to avoid hair
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
        vein_darkened = cv2.convertScaleAbs(image, alpha=0.01, beta=-255)  # Nearly black veins
        processed_frame = np.where(vein_mask, vein_darkened, processed_frame)
        processed_frame = np.clip(processed_frame, 0, 255).astype(np.uint8)
        
        # Convert back to 3-channel for visualization
        result = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
        
        return result
    
    def _frangi_filter_method(self, image):
        """Use Frangi filter to enhance tubular structures like veins"""
        # Normalize image to float
        img_float = image.astype(float) / 255.0
        
        # Apply Frangi filter to enhance tubular structures (veins)
        vein_enhanced = filters.frangi(img_float, scale_range=(1, 10), scale_step=2)
        
        # Normalize back to 0-255 range
        vein_enhanced = exposure.rescale_intensity(vein_enhanced, out_range=(0, 1))
        vein_enhanced = (vein_enhanced * 255).astype(np.uint8)
        
        # Apply threshold to segment veins
        _, binary = cv2.threshold(vein_enhanced, 15, 255, cv2.THRESH_BINARY)
        
        # Clean up the binary image
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Create an output image combining original with veins
        background = cv2.convertScaleAbs(image, alpha=1.0, beta=10)  # Brighten background
        result = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
        
        # Add veins in red
        result[:, :, 2] = cv2.max(result[:, :, 2], cleaned)  # Red channel overlay
        
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
        background = cv2.convertScaleAbs(image, alpha=1.0, beta=10)  # Brighten background
        result = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
        
        # Create a blue mask for veins
        mask = np.zeros_like(result)
        mask[:, :, 0] = cleaned  # Blue channel
        
        # Blend with original
        alpha = 0.7
        cv2.addWeighted(result, 1, mask, alpha, 0, result)
        
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