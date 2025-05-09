import pathlib, random, numpy as np, pandas as pd, cv2, tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ---------------- paths ----------------
ROOT          = pathlib.Path('.')               # project root
CROP_DIR      = ROOT / 'Cropped_Images'
CSV_PATH      = ROOT / 'true_labels.csv'

# ---------------- vars ----------------
IMG_SIZE      = 128                             # 128×128 RGB
BATCH_SIZE    = 32
EPOCHS        = 40
SEED          = 42



print('Loading label CSV …')
df = pd.read_csv(CSV_PATH)

# keep only labelled rows
df = df[df['label'] != 0].copy()
print(f'Total labelled rows in CSV: {len(df)}')

def stem_to_crop(fname):
    """IMG_1234.HEIC → IMG_1234_crop.jpg (Path object)"""
    stem = pathlib.Path(fname).stem
    return CROP_DIR / f'{stem}_crop.jpg'

df['crop_path'] = df['filename'].apply(stem_to_crop)
df = df[df['crop_path'].apply(lambda p: p.exists())]
print(f'Labelled rows with existing crop: {len(df)}')

# -----------------------------------------------------------
# split into 3 sets (70/15/15) stratified by first digit
train_df, temp_df = train_test_split(
    df, test_size=0.30, random_state=SEED, stratify=df['label']//100
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=SEED, stratify=temp_df['label']//100
)
for name, part in zip(['train','val','test'], [train_df, val_df, test_df]):
    print(f'{name:5s}: {len(part)} samples')

# -----------------------------------------------------------
def decode_label(num):
    """123 becomes [1,2,3]"""
    return [num//100, (num//10)%10, num%10]

def load_img(path):
    if isinstance(path, bytes):
        path = path.decode()    
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img.astype('float32') / 255.0

def make_dataset(df, shuffle=False):
    paths   = df['crop_path'].astype(str).values
    digits  = np.stack(df['label'].apply(decode_label).values)  # N×3
    ds      = tf.data.Dataset.from_tensor_slices((paths, digits))
    def _map(p, d):
        img = tf.numpy_function(load_img, [p], tf.float32)
        img.set_shape((IMG_SIZE, IMG_SIZE, 3))
        return img, {'d1': d[0], 'd2': d[1], 'd3': d[2]}
    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(1000, seed=SEED)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_ds = make_dataset(train_df, shuffle=True)
val_ds   = make_dataset(val_df)
test_ds  = make_dataset(test_df)

# -----------------------------------------------------------
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)

digit_outputs = [
    tf.keras.layers.Dense(10, activation='softmax', name=f'd{i+1}')(x)
    for i in range(3)
]

model = tf.keras.Model(inputs, digit_outputs)
model.compile(
    optimizer='adam',
    loss={'d1':'sparse_categorical_crossentropy',
          'd2':'sparse_categorical_crossentropy',
          'd3':'sparse_categorical_crossentropy'},
    metrics=['accuracy']
)
model.summary()

# -----------------------------------------------------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True)
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# -----------------------------------------------------------
# evaluation
results = model.evaluate(test_ds, return_dict=True)
print('\nDigit-wise accuracies:')
print(f"d1: {results['d1_accuracy']:.4f}   d2: {results['d2_accuracy']:.4f}   d3: {results['d3_accuracy']:.4f}")

# “all digits correct” accuracy
y_true = []
y_pred = []
for imgs, lbls in test_ds:
    preds = model.predict(imgs, verbose=0)
    preds = [np.argmax(p, axis=1) for p in preds]      # list of 3 arrays
    y_pred.extend(list(zip(*preds)))
    y_true.extend(list(zip(lbls['d1'].numpy(), lbls['d2'].numpy(), lbls['d3'].numpy())))
all_correct = np.mean([p==t for p,t in zip(y_pred, y_true)])
print(f'All‑3‑digits‑correct accuracy: {all_correct:.4f}')

# -----------------------------------------------------------
# visual sanity check on 8 random test images
sample_paths = random.sample(list(test_df['crop_path']), k=min(8, len(test_df)))
plt.figure(figsize=(12,6))
for i, p in enumerate(sample_paths, 1):
    img  = load_img(p)
    pred = model.predict(img[None], verbose=0)
    pred_digits = ''.join(str(np.argmax(d)) for d in pred)
    true_digits = test_df.loc[test_df['crop_path']==p, 'label'].iloc[0]
    plt.subplot(2,4,i); plt.axis('off')
    plt.title(f'pred {pred_digits}  /  true {true_digits:03d}',
              color='green' if int(pred_digits)==true_digits else 'red',
              fontsize=9)
    plt.imshow(img)
plt.tight_layout(); plt.show()
