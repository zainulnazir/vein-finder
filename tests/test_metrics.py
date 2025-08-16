#!/usr/bin/env python3
import numpy as np
import cv2
from util_metrics import compute_metrics


def test_compute_metrics_shapes():
    og = np.full((100, 100), 120, dtype=np.uint8)

    # Create a dark line to emulate a vein in processed image
    proc = np.dstack([og.copy()]*3)
    cv2.line(proc, (10, 50), (90, 50), (0, 0, 0), 3)

    m = compute_metrics(og, proc)
    assert isinstance(m, dict)
    assert 'snr' in m and 'laplacian_variance' in m and 'hist_spread' in m


def test_compute_metrics_nonempty():
    og = np.random.randint(80, 160, (64, 64), dtype=np.uint8)
    proc = np.dstack([og.copy()]*3)
    m = compute_metrics(og, proc)
    assert m['hist_spread'] >= 0
