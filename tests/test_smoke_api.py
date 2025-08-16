#!/usr/bin/env python3
import os
import threading
import time
import requests
import subprocess

BASE = "http://127.0.0.1:8001"


def wait_http(url, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(url, timeout=0.5)
            if r.status_code < 500:
                return True
        except Exception:
            pass
        time.sleep(0.2)
    return False


def test_video_and_capture_smoke():
    # Start the app in dev mode on a different port for tests
    env = os.environ.copy()
    cmd = ["python3", "app.py", "--dev", "--port", "8001"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        assert wait_http(BASE + "/test", 12.0)
        # Video feed should be reachable
        r = requests.get(BASE + "/video_feed", stream=True, timeout=4)
        assert r.status_code == 200
        # Quick capture request (adaptive) — in dev camera returns simulated frames
        r2 = requests.post(BASE + "/capture", json={"detection_method": "adaptive"}, timeout=10)
        assert r2.status_code == 200
        assert r2.json().get("success")
    finally:
        p.terminate()
        try:
            p.wait(3)
        except Exception:
            p.kill()
