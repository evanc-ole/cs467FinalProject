import json, shutil, random, pathlib, cv2, numpy as np, tqdm

SRC_JSON = pathlib.Path('RectLabels')
SRC_IMG  = pathlib.Path('To_Label')
DEST     = pathlib.Path('yolo_dataset')
DEST_IMG = DEST/'images'
DEST_LAB = DEST/'labels'
DEST_IMG.mkdir(parents=True, exist_ok=True)
DEST_LAB.mkdir(parents=True, exist_ok=True)

train_ratio = .8
files = sorted(SRC_JSON.glob('IMG_*.json'))
random.shuffle(files)

for j_path in tqdm.tqdm(files):
    split = 'train' if random.random() < train_ratio else 'val'
    img_path = SRC_IMG / pathlib.Path(j_path.stem).with_suffix('.jpg')
    img     = cv2.imread(str(img_path)); h, w = img.shape[:2]

    with open(j_path) as f: data = json.load(f)
    poly = np.array(data['shapes'][0]['points'])          # (4,2)
    poly_norm = poly / [w, h]                             # → [0‒1]

    # YOLOv8‑seg label format:  <cls> xc yc bw bh n x1 y1 x2 y2 …
    xs, ys        = poly_norm[:,0], poly_norm[:,1]
    xc, yc        = xs.mean(), ys.mean()
    bw, bh        = xs.max()-xs.min(), ys.max()-ys.min()
    flat_poly     = ' '.join(f'{x:.6f} {y:.6f}' for x, y in poly_norm)

    #y_line = f'0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f} 4 {flat_poly}\n'
    y_line = f'0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f} {flat_poly}\n'
    lab_file = DEST_LAB/split/f'{j_path.stem}.txt'
    img_file = DEST_IMG/split/img_path.name
    lab_file.parent.mkdir(parents=True, exist_ok=True)
    img_file.parent.mkdir(parents=True, exist_ok=True)

    with open(lab_file, 'w') as lf: lf.write(y_line)
    shutil.copy(img_path, img_file)
