"""
Required packages:
pip install tensorflow pillow pillow-heif

This script implements a CNN for classifying images with 1000 class outputs (000-999)
"""

import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener
import os
import argparse

# Register HEIF opener for handling HEIC files
register_heif_opener()

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Constants
IMG_SIZE = 112  # Standard input size for many CNN architectures
BATCH_SIZE = 32
NUM_CLASSES = 1000  # Labels from 000-999
EPOCHS = 20
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

def load_and_preprocess_image(image_path):
    """Load and preprocess a single image"""
    try:
        # Load image
        img = Image.open(image_path)
        
        # Convert to RGB (in case of different color spaces)
        img = img.convert('RGB')
        
        # Resize image
        img = img.resize((IMG_SIZE, IMG_SIZE))
        
        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0
        
        return img_array
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def create_dataset():
    """Create training dataset from images and labels"""
    # Read labels
    df = pd.read_csv('true_labels.csv')
    
    #BAD/NONEXISTENT FILES
    BAD_FILES = ["IMG_2835.HEIC", "IMG_2939.HEIC", "IMG_2914.HEIC"]
    
    # Remove rows with bad files
    df = df[~df['filename'].isin(BAD_FILES)]
    
    # Truncate dataset
    df = df[df.index <= df[df['filename'] == 'IMG_2947.HEIC'].index[0]]


    # Prepare data structures
    images = []
    labels = []
    
    # Process each image
    for _, row in df.iterrows():
        img_path = os.path.join('Archive', row['filename'])
        img_array = load_and_preprocess_image(img_path)
        
        if img_array is not None:
            images.append(img_array)
            labels.append(row['label'])
    
    # Convert to numpy arrays
    X = np.array(images)
    y = np.array(labels)
    
    # Convert labels to one-hot encoding
    y = tf.keras.utils.to_categorical(y, NUM_CLASSES)
    
    return X, y

def create_model():
    """Create CNN model architecture"""
    model = tf.keras.Sequential([
        # First Convolutional Block
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # Second Convolutional Block
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # Third Convolutional Block
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # Fourth Convolutional Block
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # Flatten and Dense Layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train CNN classifier with optional saving and callbacks')
    parser.add_argument('--save-final', action='store_true', help='Save the final model')
    parser.add_argument('--save-best', action='store_true', help='Save the best model during training')
    parser.add_argument('--use-callbacks', action='store_true', help='Use callbacks during training')
    args = parser.parse_args()

    # Load and preprocess data
    print("Loading and preprocessing images...")
    X, y = create_dataset()
    
    # Split data into training and validation sets
    indices = np.random.permutation(len(X))
    split_idx = int(len(X) * (1 - VALIDATION_SPLIT - TEST_SPLIT))
    split_idx_2 = int(len(X) * (1 - TEST_SPLIT))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:split_idx_2]
    test_indices = indices[split_idx_2:]
    
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    
    # Create and compile model
    print("Creating and compiling model...")
    model = create_model()
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create callbacks if requested
    callbacks = None
    if args.use_callbacks:
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            )
        ]
        if args.save_best:
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    'best_model.h5',
                    save_best_only=True,
                    monitor='val_accuracy'
                )
            )
    
    # Train model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Save final model if requested
    if args.save_final:
        model.save('final_model.h5')
        print("Model saved as 'final_model.h5'")

if __name__ == "__main__":
    main()
