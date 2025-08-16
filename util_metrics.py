#!/usr/bin/env python3

import os
import csv
import numpy as np
import cv2
from typing import Dict, Any


def compute_metrics(original_gray: np.ndarray, processed_bgr: np.ndarray) -> Dict[str, Any]:
    """Compute quantitative metrics: SNR, Laplacian variance, histogram spread.
    - original_gray: grayscale original image (H,W)
    - processed_bgr: processed image (H,W,3) or (H,W)
    Returns: dict with keys snr, laplacian_variance, hist_spread, mean_bg, mean_vein
    """
    try:
        if processed_bgr is None or original_gray is None:
            return {}
        # Normalize inputs
        if len(processed_bgr.shape) == 3 and processed_bgr.shape[2] == 3:
            processed_gray = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2GRAY)
        else:
            processed_gray = processed_bgr.copy()

        og = original_gray.astype(np.float32)
        if og.size == 0:
            return {}

        # Build vein mask from processed: detect dark ridges by thresholding inverted image (Otsu)
        inv = cv2.bitwise_not(processed_gray)
        _, th = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        vein_mask = th == 255
        bg_mask = ~vein_mask

        if vein_mask.sum() == 0 or bg_mask.sum() == 0:
            p10 = float(np.percentile(og, 10))
            p90 = float(np.percentile(og, 90))
            lap_var = float(np.var(cv2.Laplacian(og, cv2.CV_64F)))
            return {
                "snr": 0.0,
                "laplacian_variance": lap_var,
                "hist_spread": p90 - p10,
                "mean_bg": None,
                "mean_vein": None,
            }

        mean_bg = float(og[bg_mask].mean())
        std_bg = float(og[bg_mask].std()) + 1e-6
        mean_vein = float(og[vein_mask].mean())
        snr = abs(mean_bg - mean_vein) / std_bg

        lap_var = float(np.var(cv2.Laplacian(og, cv2.CV_64F)))
        p10 = float(np.percentile(og, 10))
        p90 = float(np.percentile(og, 90))

        return {
            "snr": float(snr),
            "laplacian_variance": lap_var,
            "hist_spread": float(p90 - p10),
            "mean_bg": mean_bg,
            "mean_vein": mean_vein,
        }
    except Exception as e:
        print(f"Metric computation error: {e}")
        return {}


def append_metrics_csv(save_dir: str, row_dict: Dict[str, Any]) -> None:
    """Append a row of metrics to save_dir/metrics.csv, creating header if needed."""
    try:
        csv_path = os.path.join(save_dir, "metrics.csv")
        fieldnames = [
            "timestamp",
            "detection_method",
            "exposure",
            "gain",
            "clahe_clip_limit",
            "clahe_tile_grid_size",
            "frangi_scale_min",
            "frangi_scale_max",
            "frangi_scale_step",
            "frangi_beta",
            "frangi_gamma",
            "snr",
            "laplacian_variance",
            "hist_spread",
            "mean_bg",
            "mean_vein",
        ]
        file_exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({k: row_dict.get(k, "") for k in fieldnames})
    except Exception as e:
        print(f"CSV append error: {e}")
