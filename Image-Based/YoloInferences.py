from ultralytics import YOLO
import pathlib, random, cv2, numpy as np

LABEL_DIR = pathlib.Path('RectLabels') 
IMG_DIR   = pathlib.Path('To_Label')
OUT_DIR   = pathlib.Path('Yolo_inferences_results')
OUT_DIR.mkdir(exist_ok=True)

# ---------------- find unlabelled images (test data) ----------------
labelled_stems = {p.stem for p in LABEL_DIR.glob('*.json')}

all_imgs     = list(IMG_DIR.glob('*.jpg'))
labelled_imgs = [p for p in all_imgs if p.stem in labelled_stems]
test_imgs     = [p for p in all_imgs if p.stem not in labelled_stems]

#print(f'Labelled images  : {len(labelled_imgs)}')
print(f'Unlabelled images: {len(test_imgs)}')

# ---------------- pick 5 random test images ----------------
random.seed(42)
sampled = random.sample(test_imgs, min(5, len(test_imgs)))


model = YOLO('runs/segment/train/weights/best.pt')

def quad_from_pred(im):
    res   = model.predict(im, conf=0.25, verbose=False)[0]
    if not res.masks: 
        print('cmon')
        return None

    # take highest‑conf mask (prob 1 per image)
    mask  = res.masks.xy[0]              # (N,2) polygon in pixel coords
    rect  = cv2.minAreaRect(mask.astype(np.float32))
    box   = cv2.boxPoints(rect)          # (4,2)
    return np.intp(box)

# ---------------- inferences ----------------
for img_path in sampled:
    img  = cv2.imread(str(img_path))
    quad = quad_from_pred(img)

    if quad is not None:
        cv2.polylines(img, [quad], True, (0, 255, 0), 3)
    else:
        print(f'No rectangle detected in {img_path.name}')

    out_file = OUT_DIR / img_path.name
    cv2.imwrite(str(out_file), img)

print('✓ Done - results written to', OUT_DIR)