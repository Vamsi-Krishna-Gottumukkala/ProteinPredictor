# --- [START OF FIX] ---
# CRITICAL PATCH for Python 3.12 + TensorFlow
import sys
try:
    import distutils
except ImportError:
    import setuptools
    sys.modules['distutils'] = setuptools._distutils
# --- [END OF FIX] ---

import numpy as np
import pandas as pd
import tensorflow as tf
import os

from src.predictors import get_distance_prediction_model, one_hot_encode

# CONFIG
MAX_SEQ_LEN = 100
CSV_PATH = "dataset/distance_data.csv"
MODEL_SAVE_PATH = "models/distance_weights.weights.h5"


def load_distance_data():
    """Load training data with sequences and distance matrices."""
    if not os.path.exists(CSV_PATH):
        print("❌ Error: dataset/distance_data.csv not found.")
        print("   Run 'python dataset/fetch_real_data.py' first!")
        return None, None
    
    df = pd.read_csv(CSV_PATH)
    X = []  # Input: Sequences
    Y = []  # Output: Distance matrices
    
    for i, row in df.iterrows():
        seq = row['sequence']
        dist_file = row['distance_file']
        
        # Skip if sequence too long
        if len(seq) > MAX_SEQ_LEN:
            continue
        
        # Load distance matrix
        if not os.path.exists(dist_file):
            continue
        
        dist_matrix = np.load(dist_file)
        
        # Prepare One-Hot Input
        x_enc = one_hot_encode(seq, MAX_SEQ_LEN)[0]
        X.append(x_enc)
        
        # Prepare Target Distance Matrix (pad to MAX_SEQ_LEN x MAX_SEQ_LEN)
        seq_len = len(seq)
        y_dist = np.zeros((MAX_SEQ_LEN, MAX_SEQ_LEN))
        y_dist[:seq_len, :seq_len] = dist_matrix[:seq_len, :seq_len]
        Y.append(y_dist)
    
    if len(X) == 0:
        print("❌ No valid samples found!")
        return None, None
    
    return np.array(X), np.array(Y)


def train():
    print("=" * 50)
    print("  Distance Prediction Model - Training")
    print("  Architecture: 1D-CNN → 2D-CNN")
    print("=" * 50)
    
    print("\n--- 1. Loading Data ---")
    X, Y = load_distance_data()
    if X is None:
        return
    
    print(f"✅ Loaded {len(X)} protein samples for training.")
    print(f"   Input shape: {X.shape}")
    print(f"   Distance matrix shape: {Y.shape}")
    
    # Distance statistics
    nonzero_distances = Y[Y > 0]
    print(f"\n   Distance Statistics:")
    print(f"   - Min: {nonzero_distances.min():.2f} Å")
    print(f"   - Max: {nonzero_distances.max():.2f} Å")
    print(f"   - Mean: {nonzero_distances.mean():.2f} Å")
    
    print("\n--- 2. Building Distance Prediction Model ---")
    model = get_distance_prediction_model(MAX_SEQ_LEN)
    model.summary()
    
    print("\n--- 3. Training ---")
    
    # Callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )
    
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-6
    )
    
    history = model.fit(
        X, Y,
        epochs=100,
        batch_size=2,  # Small batch due to L×L output
        validation_split=0.2,
        callbacks=[early_stop, lr_scheduler],
        verbose=1
    )
    
    print("\n--- 4. Saving Model Weights ---")
    os.makedirs("models", exist_ok=True)
    model.save_weights(MODEL_SAVE_PATH)
    print(f"✅ Model weights saved to: {MODEL_SAVE_PATH}")
    
    # Print final metrics
    print("\n--- Training Complete ---")
    print(f"Final Loss (MSE): {history.history['loss'][-1]:.4f}")
    print(f"Final MAE: {history.history['mae'][-1]:.4f} Å")


if __name__ == "__main__":
    train()
