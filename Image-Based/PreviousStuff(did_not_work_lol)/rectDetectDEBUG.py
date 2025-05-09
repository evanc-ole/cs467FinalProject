#!/usr/bin/env python3
"""
Crop the black‑background, white‑digit panel from warehouse stickers.

Debug‑mode saves intermediate images so you can see what to tune.
"""

# --------- dependencies ---------
# pip install opencv-python imageio[ffmpeg] numpy tqdm
import cv2, imageio.v3 as iio, numpy as np, sys, pathlib, itertools
from tqdm import tqdm

# ======= KNOBS YOU WILL TWEAK MOST OFTEN =======
AREA_MIN_FRAC, AREA_MAX_FRAC = 0.0002, 0.2   # candidate blob size (rel. to frame)
ASPECT_MIN,   ASPECT_MAX    = 1.0,  3.0     # sticker is ~2:1 (w:h)
BLUR_KSIZE                   = (5, 5)       # Gaussian‑blur kernel
MORPH_KSIZE                  = (5, 5)       # closing kernel
# ===============================================

SRC_DIR = pathlib.Path("../Archive")
CROP_DIR = pathlib.Path("../Crops");   CROP_DIR.mkdir(exist_ok=True)
DBG_DIR  = pathlib.Path("../Debug");   DBG_DIR.mkdir(exist_ok=True)

def dump(stage, stem, img):
    cv2.imwrite(str(DBG_DIR / f"{stem}_{stage}.png"), img)

def process_one(img_path, debug=True):
    stem = img_path.stem
    # 0 ─ read HEIC
    rgb = iio.imread(img_path)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # 1 ─ grayscale → equalise → invert
    g   = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g   = cv2.equalizeHist(g);                dump("gray_eq", stem, g)       if debug else None
    g   = 255 - g;                            dump("invert",  stem, g)       if debug else None

    # 2 ─ blur
    g   = cv2.GaussianBlur(g, BLUR_KSIZE, 0); dump("blur", stem, g)          if debug else None

    # 3 ─ threshold (Otsu)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dump("thresh", stem, th)                 if debug else None

    # 4 ─ morphology close
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KSIZE)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    dump("close", stem, th)                  if debug else None

    # 5 ─ contours  ▸ area ▸ aspect filters
    h, w = th.shape
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in cnts:
        area_frac = cv2.contourArea(c) / (h * w)
        if not AREA_MIN_FRAC < area_frac < AREA_MAX_FRAC:
            continue
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(c)
        if rw == 0 or rh == 0: continue
        aspect = max(rw, rh) / min(rw, rh)
        if ASPECT_MIN <= aspect <= ASPECT_MAX:
            candidates.append((area_frac, (cx, cy), (rw, rh), ang, c))

    if not candidates:
        print(f"⚠️  {stem}: no candidate rectangle");  return False

    # 6 ─ take biggest
    _, center, size, angle, best_cnt = max(candidates, key=lambda t: t[0])

    # 7 ─ visualise contours
    dbg_ct = bgr.copy()
    for _,_,_,_,c in candidates:
        cv2.drawContours(dbg_ct, [cv2.boxPoints(cv2.minAreaRect(c)).astype(int)], -1, (0,255,0), 2)
    dump("contours", stem, dbg_ct)           if debug else None

    # 8 ─ deskew
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    warped = cv2.warpAffine(bgr, M, (w, h), flags=cv2.INTER_CUBIC)

    # 9 ─ axis‑aligned crop
    box = cv2.boxPoints(((center), size, angle))            # original
    box = cv2.transform(np.array([box]), M)[0]              # rotated
    x,y,wc,hc = cv2.boundingRect(box.astype(int))
    crop = warped[y:y+hc, x:x+wc]
    cv2.imwrite(str(CROP_DIR / f"{stem}.png"), crop)
    dump("crop", stem, crop)                if debug else None
    return True


# ---------- main ----------
args = sys.argv[1:]
images = [SRC_DIR / a for a in args] if args else sorted(SRC_DIR.glob("IMG_*.HEIC"))
success = 0
for p in tqdm(images, desc="Cropping"):
    try:
        success += process_one(p, debug=True)
    except Exception as e:
        print(f"⚠️  {p.stem}: {e}")

print(f"\nFinished.  {success}/{len(images)} succeeded.")
print("Inspect the 'debug/' folder ➜ tweak KNOBS at top ➜ run again.")
