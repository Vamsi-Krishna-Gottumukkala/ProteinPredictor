import numpy as np
import pandas as pd
import tensorflow as tf
import os
import sys

# Patch for Python 3.12 (Just in case)
try:
    import distutils
except ImportError:
    import setuptools
    sys.modules['distutils'] = setuptools._distutils

from src.predictors import get_cnn_bilstm_model, one_hot_encode, ss_to_one_hot, SS_MAP

# CONFIG
MAX_SEQ_LEN = 100
CSV_PATH = "dataset/train_data.csv"
MODEL_SAVE_PATH = "models/cnn_bilstm_weights.weights.h5"
LEGACY_MODEL_PATH = "models/cnn_weights.weights.h5"

def load_data():
    """Load training data with sequences, angles, and secondary structure labels."""
    if not os.path.exists(CSV_PATH):
        print("❌ Error: dataset/train_data.csv not found. Run dataset/fetch_real_data.py first!")
        return None, None, None
        
    df = pd.read_csv(CSV_PATH)
    X = []  # Input: Sequences
    Y_angles = []  # Output: Angles
    Y_ss = []  # Output: Secondary Structure
    
    # Check if secondary_structure column exists
    has_ss = 'secondary_structure' in df.columns
    if not has_ss:
        print("⚠️ Warning: No secondary_structure column found. Using angles to infer SS.")
    
    for i, row in df.iterrows():
        seq = row['sequence']
        angles_str = row['angles']
        
        # Parse angles string back to list
        try:
            angles = [float(a) for a in angles_str.split()]
        except:
            continue
            
        # Filter: Only use sequences that fit our model window
        if len(seq) > MAX_SEQ_LEN:
            seq = seq[:MAX_SEQ_LEN]
            angles = angles[:MAX_SEQ_LEN * 2]  # 2 angles per residue
        
        # Prepare One-Hot Input
        x_enc = one_hot_encode(seq, MAX_SEQ_LEN)[0] 
        X.append(x_enc)
        
        # Prepare Target Angles (Pad with zeros if short)
        y_angles = np.zeros((MAX_SEQ_LEN, 2))
        for j in range(len(seq)):
            if j*2+1 < len(angles):
                y_angles[j, 0] = angles[j*2]     # Phi
                y_angles[j, 1] = angles[j*2+1]   # Psi
        Y_angles.append(y_angles)
        
        # Prepare Target Secondary Structure
        if has_ss:
            ss_string = row['secondary_structure']
            # Pad or truncate to MAX_SEQ_LEN
            ss_string = ss_string[:MAX_SEQ_LEN].ljust(MAX_SEQ_LEN, 'C')
            y_ss = ss_to_one_hot(ss_string, MAX_SEQ_LEN)
        else:
            # Infer from angles
            y_ss = np.zeros((MAX_SEQ_LEN, 3))
            for j in range(len(seq)):
                if j*2+1 < len(angles):
                    phi, psi = angles[j*2], angles[j*2+1]
                    # Simple classification based on angle regions
                    if -0.5 < phi < -0.2 and -0.4 < psi < -0.1:
                        y_ss[j] = [1, 0, 0]  # Helix
                    elif phi < -0.5 and (psi > 0.5 or psi < -0.7):
                        y_ss[j] = [0, 1, 0]  # Sheet
                    else:
                        y_ss[j] = [0, 0, 1]  # Coil
                else:
                    y_ss[j] = [0, 0, 1]  # Default to coil
        Y_ss.append(y_ss)
        
    return np.array(X), np.array(Y_ss), np.array(Y_angles)

def train():
    print("=" * 50)
    print("  Protein Structure Prediction - Training")
    print("  Architecture: 1D-CNN + Bi-LSTM")
    print("=" * 50)
    
    print("\n--- 1. Loading Data ---")
    X, Y_ss, Y_angles = load_data()
    if X is None: 
        return
    
    print(f"✅ Loaded {len(X)} protein sequences for training.")
    print(f"   Input shape: {X.shape}")
    print(f"   SS output shape: {Y_ss.shape}")
    print(f"   Angle output shape: {Y_angles.shape}")
    
    # Calculate SS distribution
    ss_counts = np.sum(Y_ss, axis=(0, 1))
    total = np.sum(ss_counts)
    print(f"\n   Secondary Structure Distribution:")
    print(f"   - Helix (H): {ss_counts[0]:.0f} ({100*ss_counts[0]/total:.1f}%)")
    print(f"   - Sheet (E): {ss_counts[1]:.0f} ({100*ss_counts[1]/total:.1f}%)")
    print(f"   - Coil (C): {ss_counts[2]:.0f} ({100*ss_counts[2]/total:.1f}%)")
    
    print("\n--- 2. Building CNN + Bi-LSTM Model ---")
    model = get_cnn_bilstm_model(MAX_SEQ_LEN)
    model.summary()
    
    print("\n--- 3. Training (Dual-Task Learning) ---")
    
    # Early stopping callback
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Learning rate scheduler
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    
    history = model.fit(
        X, 
        {'ss_output': Y_ss, 'angle_output': Y_angles},
        epochs=50,
        batch_size=4,
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
    print(f"Final SS Accuracy: {history.history['ss_output_accuracy'][-1]:.4f}")
    print(f"Final Angle MAE: {history.history['angle_output_mae'][-1]:.4f}")

if __name__ == "__main__":
    train()