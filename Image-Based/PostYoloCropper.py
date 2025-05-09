import pathlib, cv2, numpy as np
from ultralytics import YOLO

# ---------------- paths ----------------
IMG_DIR = pathlib.Path('To_Label')
OUT_DIR = pathlib.Path('Cropped_Images')
OUT_DIR.mkdir(exist_ok=True)

# ---------------- load model ----------------
model = YOLO('runs/segment/train/weights/best.pt')

def quad_from_pred(im):
    """Return 4x2 int array (clockwise) or None if nothing detected."""
    res = model.predict(im, conf=0.25, verbose=False)[0]
    if not res.masks:
        print('model inference tweeeaaakking yoo')
        return None
    poly = res.masks.xy[0]                  # N×2 polygon
    rect = cv2.minAreaRect(poly.astype(np.float32))
    box  = cv2.boxPoints(rect)              # 4×2
    # order corners: TL, TR, BR, BL
    s = box.sum(1);  diff = np.diff(box, axis=1).flatten()
    ordered = np.array([box[s.argmin()],      # tl  (smallest x+y)
                        box[diff.argmin()],   # tr  (smallest x‑y)
                        box[s.argmax()],      # br  (largest x+y)
                        box[diff.argmax()]])  # bl  (largest x‑y)
    return np.intp(ordered)

def crop_by_quad(img, quad):
    (tl, tr, br, bl) = quad
    # compute width & height of the new image
    width  = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst    = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype="float32")
    M      = cv2.getPerspectiveTransform(quad.astype("float32"), dst)
    return cv2.warpPerspective(img, M, (width, height))

# --- For all images, use model toinfer the bounding box and crop the image ---
for img_path in IMG_DIR.glob('*.jpg'):
    img  = cv2.imread(str(img_path))
    quad = quad_from_pred(img)

    if quad is None:
        print(f'[!] No rectangle in {img_path.name}')
        continue

    cropped = crop_by_quad(img, quad)
    out_name = OUT_DIR / f'{img_path.stem}_crop.jpg'
    cv2.imwrite(str(out_name), cropped)

print('✓ Cropping complete - results saved to', OUT_DIR)
