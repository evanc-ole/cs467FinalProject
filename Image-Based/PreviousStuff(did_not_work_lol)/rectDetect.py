#!/usr/bin/env python3
# classical_crop_digits.py
#
# Usage:
#   pip install opencv-python imageio[ffmpeg] tqdm numpy
#   python classical_crop_digits.py
#
# Result:
#   crops/IMG_0123.png, crops/IMG_0124.png, ...
#   plus a list of any images that could not be cropped automatically.

import cv2
import imageio.v3 as iio          # reads HEIC out‑of‑the‑box if libheif is installed
import numpy as np
import pathlib, sys
from tqdm import tqdm

SRC_DIR   = pathlib.Path("../Archive")
DST_DIR   = pathlib.Path("../Crops")
DST_DIR.mkdir(exist_ok=True)

# ---- hyper‑parameters you might tune later ----
AREA_MIN_FRAC   = 0.02            # ignore blobs smaller than  2 % of the image
AREA_MAX_FRAC   = 0.80            # ignore blobs larger than 80 % (e.g. whole frame)
ASPECT_MIN      = 2.0             # sticker is ~3:1, so accept 2 – 4
ASPECT_MAX      = 4.0

failures = []

for img_path in tqdm(sorted(SRC_DIR.glob("IMG_*.HEIC"))):
    try:
        # 0) --- read & convert HEIC ➜ BGR numpy array ---
        img  = iio.imread(img_path)                # RGB uint8
        img  = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # 1) --- basic preprocessing ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)              # rescues low‑contrast shots
        gray = 255 - gray                          # invert: black rectangle ➜ white,
                                                   #           white digits ➜ dark
        gray = cv2.GaussianBlur(gray, (5, 5), 0)   # suppress salt‑and‑pepper noise

        # 2) --- binarise (white rectangle vs. dark background) ---
        _, bin_img = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )

        # 3) --- close small gaps in the rectangle border ---
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)

        # 4) --- find candidate contours ---
        cnts, _ = cv2.findContours(
            bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # 5) --- filter by area and aspect ratio ---
        h, w = bin_img.shape
        candidates = []
        for c in cnts:
            area = cv2.contourArea(c) / (h * w)
            if area < AREA_MIN_FRAC or area > AREA_MAX_FRAC:
                continue
            rect = cv2.minAreaRect(c)              # ((cx,cy),(w,h),θ)
            aw, ah = rect[1]
            if aw == 0 or ah == 0:
                continue
            aspect = max(aw, ah) / min(aw, ah)
            if ASPECT_MIN <= aspect <= ASPECT_MAX:
                candidates.append((area, rect))

        if not candidates:
            raise RuntimeError("no rectangle-like contour found")

        # 6) --- take the largest surviving rectangle ---
        _, best = max(candidates, key=lambda t: t[0])

        # 7) --- deskew the sticker so the crop is axis‑aligned ---
        (cx, cy), (rw, rh), angle = best
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        warped = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)

        # 8) --- find axis‑aligned bounding box of the rotated contour ---
        box_pts = cv2.boxPoints(best)
        box_pts = cv2.transform(np.array([box_pts]), M)[0]
        x, y, bw, bh = cv2.boundingRect(box_pts.astype(np.int32))

        roi = warped[y : y + bh, x : x + bw]

        # 9) --- save crop as PNG ---
        out_name = DST_DIR / (img_path.stem + ".png")
        cv2.imwrite(str(out_name), roi)

    except Exception as e:
        failures.append((img_path.name, str(e)))

# --- report ---
if failures:
    print("\n⚠️  Could not crop the following images:")
    for fname, msg in failures:
        print(f"  {fname}: {msg}")
else:
    print("\n✅  All images cropped successfully.")
