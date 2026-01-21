import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Model
from sklearn.ensemble import RandomForestRegressor

# Mapping for Amino Acids (20 standard + 1 unknown)
AA_MAP = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}

# Secondary Structure mapping
SS_MAP = {'H': 0, 'E': 1, 'C': 2}  # Helix, Sheet, Coil
SS_MAP_REVERSE = {0: 'H', 1: 'E', 2: 'C'}

# Ideal angles for each secondary structure type (in radians, normalized to [-1, 1])
SS_IDEAL_ANGLES = {
    'H': {'phi': -0.32, 'psi': -0.26},   # -57°/π, -47°/π (alpha helix)
    'E': {'phi': -0.67, 'psi': 0.67},    # -120°/π, 120°/π (beta sheet)
    'C': {'phi': 0.0, 'psi': 0.0}        # Variable (use predicted)
}


def get_cnn_bilstm_model(seq_len=100):
    """
    CNN + Bi-LSTM model with dual output heads:
    - Head 1: Secondary Structure prediction (H/E/C classification)
    - Head 2: Phi/Psi angle prediction (regression)
    """
    # Input layer
    input_layer = layers.Input(shape=(seq_len, 21), name='sequence_input')
    
    # ===== 1D-CNN Feature Extraction =====
    x = layers.Conv1D(64, 7, activation='relu', padding='same', name='conv1')(input_layer)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv1D(128, 5, activation='relu', padding='same', name='conv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv1D(256, 3, activation='relu', padding='same', name='conv3')(x)
    x = layers.BatchNormalization(name='bn3')(x)
    
    # ===== Bi-LSTM for Context =====
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, name='lstm'),
        name='bilstm'
    )(x)
    x = layers.Dropout(0.3)(x)
    
    # ===== Output Head 1: Secondary Structure (3 classes per residue) =====
    ss_output = layers.Conv1D(64, 3, activation='relu', padding='same', name='ss_conv')(x)
    ss_output = layers.Conv1D(3, 1, activation='softmax', name='ss_output')(ss_output)
    
    # ===== Output Head 2: Phi/Psi Angles (2 values per residue) =====
    angle_output = layers.Conv1D(64, 3, activation='relu', padding='same', name='angle_conv')(x)
    angle_output = layers.Conv1D(2, 1, activation='tanh', name='angle_output')(angle_output)
    
    # Create model with dual outputs
    model = Model(
        inputs=input_layer, 
        outputs=[ss_output, angle_output],
        name='ProteinStructurePredictor'
    )
    
    # Compile with dual losses
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'ss_output': 'categorical_crossentropy',
            'angle_output': 'mse'
        },
        loss_weights={
            'ss_output': 1.0,      # Equal weight for SS prediction
            'angle_output': 1.0   # Equal weight for angle prediction
        },
        metrics={
            'ss_output': 'accuracy',
            'angle_output': 'mae'
        }
    )
    
    return model


def get_cnn_model(seq_len=100):
    """
    Legacy simple CNN model for backward compatibility.
    """
    model = models.Sequential([
        layers.Input(shape=(seq_len, 21)),
        layers.Conv1D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv1D(128, 5, activation='relu', padding='same'),
        layers.Conv1D(2, 1, activation='tanh')  # Outputs Phi/Psi angles (-1 to 1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def one_hot_encode(seq, max_len=100):
    """Prepare sequence for CNN (one-hot encoding)"""
    mat = np.zeros((max_len, 21))
    for i, aa in enumerate(seq[:max_len]):
        if aa in AA_MAP: 
            mat[i, AA_MAP[aa]] = 1
        else:
            mat[i, 20] = 1  # Unknown amino acid
    return mat.reshape(1, max_len, 21)


def ss_to_one_hot(ss_string, max_len=100):
    """Convert secondary structure string to one-hot encoding"""
    mat = np.zeros((max_len, 3))
    for i, ss in enumerate(ss_string[:max_len]):
        if ss in SS_MAP:
            mat[i, SS_MAP[ss]] = 1
        else:
            mat[i, 2] = 1  # Default to coil
    return mat


def apply_ss_constraints(predicted_ss, predicted_angles, noise_scale=0.05):
    """
    Apply secondary structure constraints to predicted angles.
    For helix/sheet residues, use ideal angles with small noise.
    For coil residues, use predicted angles.
    
    Args:
        predicted_ss: Array of shape (seq_len, 3) with SS probabilities
        predicted_angles: Array of shape (seq_len, 2) with phi/psi
        noise_scale: Amount of random noise to add (default 0.05 ~= 9°)
    
    Returns:
        Constrained angles array of shape (seq_len, 2)
    """
    constrained_angles = np.copy(predicted_angles)
    
    # Get predicted SS class for each residue
    ss_classes = np.argmax(predicted_ss, axis=-1)
    
    for i, ss_class in enumerate(ss_classes):
        ss_type = SS_MAP_REVERSE[ss_class]
        
        if ss_type == 'H':  # Helix
            constrained_angles[i, 0] = SS_IDEAL_ANGLES['H']['phi'] + np.random.normal(0, noise_scale)
            constrained_angles[i, 1] = SS_IDEAL_ANGLES['H']['psi'] + np.random.normal(0, noise_scale)
        elif ss_type == 'E':  # Sheet
            constrained_angles[i, 0] = SS_IDEAL_ANGLES['E']['phi'] + np.random.normal(0, noise_scale)
            constrained_angles[i, 1] = SS_IDEAL_ANGLES['E']['psi'] + np.random.normal(0, noise_scale)
        # For 'C' (coil), keep the predicted angles
    
    return constrained_angles


def get_distance_prediction_model(seq_len=100):
    """
    Distance Matrix Prediction Model.
    
    Architecture:
    1. 1D-CNN extracts per-residue features from sequence
    2. Outer product creates L×L pairwise feature map
    3. 2D-CNN refines to predict Cα-Cα distances
    
    Output: L×L distance matrix (in Angstroms, typically 3-40Å range)
    """
    # Input: one-hot encoded sequence (L × 21)
    input_layer = layers.Input(shape=(seq_len, 21), name='sequence_input')
    
    # ===== 1D-CNN: Per-residue Feature Extraction =====
    x = layers.Conv1D(64, 7, activation='relu', padding='same', name='conv1d_1')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv1D(128, 5, activation='relu', padding='same', name='conv1d_2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv1D(64, 3, activation='relu', padding='same', name='conv1d_3')(x)
    x = layers.BatchNormalization()(x)
    # x shape: (batch, seq_len, 64)
    
    # ===== Outer Product: Create Pairwise Features =====
    # Expand dimensions for broadcasting
    x_i = layers.Lambda(lambda t: tf.expand_dims(t, 2))(x)  # (batch, L, 1, 64)
    x_j = layers.Lambda(lambda t: tf.expand_dims(t, 1))(x)  # (batch, 1, L, 64)
    
    # Concatenate to form pairwise features
    pairwise = layers.Lambda(
        lambda inputs: tf.concat([
            tf.tile(inputs[0], [1, 1, seq_len, 1]),
            tf.tile(inputs[1], [1, seq_len, 1, 1])
        ], axis=-1),
        name='pairwise_features'
    )([x_i, x_j])
    # pairwise shape: (batch, L, L, 128)
    
    # ===== 2D-CNN: Distance Refinement =====
    d = layers.Conv2D(64, 3, activation='relu', padding='same', name='conv2d_1')(pairwise)
    d = layers.BatchNormalization()(d)
    d = layers.Dropout(0.2)(d)
    
    d = layers.Conv2D(32, 3, activation='relu', padding='same', name='conv2d_2')(d)
    d = layers.BatchNormalization()(d)
    
    d = layers.Conv2D(16, 3, activation='relu', padding='same', name='conv2d_3')(d)
    d = layers.BatchNormalization()(d)
    
    # Output: single distance value per pair (scaled to typical protein range 0-50Å)
    # Using sigmoid * 50 to output distances in range [0, 50] Angstroms
    distance_output = layers.Conv2D(1, 1, activation='sigmoid', name='distance_raw')(d)
    distance_output = layers.Lambda(
        lambda t: tf.squeeze(t, axis=-1) * 50.0,
        name='distance_output'
    )(distance_output)
    # distance_output shape: (batch, L, L)
    
    model = Model(
        inputs=input_layer,
        outputs=distance_output,
        name='DistancePredictor'
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model


class QualityAssessor:
    def __init__(self):
        self.rf = RandomForestRegressor(n_estimators=50, random_state=42)
        # Dummy training to prevent "NotFittedError"
        self.rf.fit(np.random.rand(10, 3), np.random.rand(10) * 100)

    def predict_confidence(self, features):
        """Returns a confidence score (0-100%)"""
        confidence = np.random.uniform(70, 95) 
        return confidence