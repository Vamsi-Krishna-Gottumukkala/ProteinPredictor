"""
Quality Assessment Module for Protein Structure Prediction

This module evaluates the physical correctness of predicted protein structures
and provides per-residue confidence scores using Random Forest regression.

Features analyzed:
- Atomic clashes (steric overlaps)
- Bond length deviations
- Bond angle deviations
- Hydrophobic packing score
- Radius of gyration

Output: Per-residue confidence scores (0-100%)
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from typing import Tuple, List, Dict

# Amino acid properties
HYDROPHOBIC_AA = set('AILMFVPGW')
POLAR_AA = set('STNQCY')
CHARGED_AA = set('DEKRH')

# Ideal geometric parameters (in Angstroms and degrees)
IDEAL_CA_CA_DISTANCE = 3.8  # Between consecutive Cα atoms
IDEAL_CA_CA_TOLERANCE = 0.5  # Acceptable deviation
MIN_NONBONDED_DISTANCE = 3.0  # Minimum distance for non-bonded atoms
IDEAL_BACKBONE_ANGLE = 110.0  # Approximate N-CA-C angle


class QualityAssessmentModule:
    """
    Evaluates structural quality and provides confidence scores.
    
    This module implements the QA pipeline as described in the project:
    1. Analyze physical properties (clashes, bonds, packing)
    2. Use Random Forest to estimate RMSD
    3. Convert to per-residue confidence scores
    """
    
    def __init__(self):
        """Initialize the QA module with a pre-trained Random Forest."""
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        # Pre-train with synthetic data (will be replaced with real training)
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize Random Forest with synthetic training data."""
        # Generate synthetic training data
        # Features: [clash_score, bond_deviation, angle_deviation, hydrophobic_score, rg]
        np.random.seed(42)
        n_samples = 500
        
        # Simulate various quality levels
        X_train = []
        y_train = []
        
        for _ in range(n_samples):
            # Good structures: low clashes, low deviations
            if np.random.random() < 0.3:
                clash = np.random.uniform(0, 0.1)
                bond_dev = np.random.uniform(0, 0.3)
                angle_dev = np.random.uniform(0, 10)
                hydro = np.random.uniform(0.5, 1.0)
                rg = np.random.uniform(10, 20)
                rmsd = np.random.uniform(1, 3)  # Low RMSD = good
            # Medium structures
            elif np.random.random() < 0.6:
                clash = np.random.uniform(0.1, 0.3)
                bond_dev = np.random.uniform(0.3, 0.6)
                angle_dev = np.random.uniform(10, 25)
                hydro = np.random.uniform(0.3, 0.6)
                rg = np.random.uniform(15, 30)
                rmsd = np.random.uniform(3, 6)
            # Poor structures
            else:
                clash = np.random.uniform(0.3, 1.0)
                bond_dev = np.random.uniform(0.6, 1.5)
                angle_dev = np.random.uniform(25, 45)
                hydro = np.random.uniform(0, 0.4)
                rg = np.random.uniform(25, 50)
                rmsd = np.random.uniform(6, 15)
            
            X_train.append([clash, bond_dev, angle_dev, hydro, rg])
            y_train.append(rmsd)
        
        self.rf_model.fit(np.array(X_train), np.array(y_train))
    
    def detect_atomic_clashes(self, coords: np.ndarray) -> Tuple[float, List[Tuple[int, int]]]:
        """
        Detect steric clashes (atoms too close together).
        
        Args:
            coords: (N, 3) array of atom coordinates (N, CA, C backbone)
            
        Returns:
            clash_score: Fraction of clashing atom pairs (0-1)
            clash_pairs: List of (residue_i, residue_j) pairs that clash
        """
        n_atoms = len(coords)
        clash_pairs = []
        total_pairs = 0
        
        # Check all non-bonded atom pairs
        for i in range(n_atoms):
            for j in range(i + 4, n_atoms):  # Skip bonded neighbors
                dist = np.linalg.norm(coords[i] - coords[j])
                total_pairs += 1
                
                if dist < MIN_NONBONDED_DISTANCE:
                    # Convert atom indices to residue indices
                    res_i = i // 3
                    res_j = j // 3
                    clash_pairs.append((res_i, res_j))
        
        clash_score = len(clash_pairs) / max(total_pairs, 1)
        return clash_score, clash_pairs
    
    def check_bond_lengths(self, coords: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Check Cα-Cα bond length deviations from ideal (3.8Å).
        
        Args:
            coords: (N, 3) array of backbone atom coordinates
            
        Returns:
            avg_deviation: Average deviation from ideal bond length
            per_residue_deviation: Deviation for each residue
        """
        # Extract Cα atoms (every 3rd atom starting from index 1)
        ca_indices = list(range(1, len(coords), 3))
        ca_coords = coords[ca_indices]
        
        n_residues = len(ca_coords)
        deviations = []
        
        for i in range(n_residues - 1):
            dist = np.linalg.norm(ca_coords[i+1] - ca_coords[i])
            deviation = abs(dist - IDEAL_CA_CA_DISTANCE)
            deviations.append(deviation)
        
        # Last residue gets same deviation as second-to-last
        if deviations:
            deviations.append(deviations[-1])
        else:
            deviations = [0.0]
        
        avg_deviation = np.mean(deviations)
        return avg_deviation, np.array(deviations)
    
    def check_bond_angles(self, coords: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Check backbone bond angle deviations.
        
        Analyzes the N-CA-C angles for each residue.
        
        Args:
            coords: (N, 3) array of backbone coordinates
            
        Returns:
            avg_deviation: Average angle deviation in degrees
            per_residue_deviation: Deviation for each residue
        """
        n_residues = len(coords) // 3
        deviations = []
        
        for i in range(n_residues):
            n_idx = i * 3
            ca_idx = i * 3 + 1
            c_idx = i * 3 + 2
            
            # Vectors
            v1 = coords[n_idx] - coords[ca_idx]
            v2 = coords[c_idx] - coords[ca_idx]
            
            # Calculate angle
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.degrees(np.arccos(cos_angle))
            
            deviation = abs(angle - IDEAL_BACKBONE_ANGLE)
            deviations.append(deviation)
        
        avg_deviation = np.mean(deviations) if deviations else 0.0
        return avg_deviation, np.array(deviations)
    
    def calculate_hydrophobic_score(self, coords: np.ndarray, sequence: str) -> Tuple[float, np.ndarray]:
        """
        Calculate hydrophobic packing score.
        
        Measures how well hydrophobic residues are buried in the core.
        
        Args:
            coords: Backbone coordinates
            sequence: Amino acid sequence
            
        Returns:
            global_score: Overall hydrophobic packing (0-1, higher is better)
            per_residue_score: Packing score for each residue
        """
        n_residues = len(sequence)
        ca_indices = list(range(1, len(coords), 3))
        ca_coords = coords[ca_indices[:n_residues]]
        
        # Calculate centroid
        centroid = np.mean(ca_coords, axis=0)
        
        # Calculate distance from centroid for each residue
        distances = np.linalg.norm(ca_coords - centroid, axis=1)
        
        # Normalize distances
        max_dist = np.max(distances) + 1e-10
        normalized_dist = distances / max_dist
        
        per_residue_score = []
        hydrophobic_burial = []
        
        for i, aa in enumerate(sequence):
            if aa in HYDROPHOBIC_AA:
                # Hydrophobic residues should be buried (low distance from center)
                burial = 1.0 - normalized_dist[i]
                hydrophobic_burial.append(burial)
                per_residue_score.append(burial)
            elif aa in POLAR_AA or aa in CHARGED_AA:
                # Polar/charged should be exposed (high distance from center)
                exposure = normalized_dist[i]
                per_residue_score.append(exposure)
            else:
                per_residue_score.append(0.5)
        
        # Global score is average of hydrophobic burial
        global_score = np.mean(hydrophobic_burial) if hydrophobic_burial else 0.5
        
        return global_score, np.array(per_residue_score)
    
    def calculate_radius_of_gyration(self, coords: np.ndarray) -> float:
        """
        Calculate radius of gyration (measure of compactness).
        
        Args:
            coords: Atom coordinates
            
        Returns:
            Radius of gyration in Angstroms
        """
        centroid = np.mean(coords, axis=0)
        distances_sq = np.sum((coords - centroid) ** 2, axis=1)
        rg = np.sqrt(np.mean(distances_sq))
        return rg
    
    def estimate_rmsd(self, features: np.ndarray) -> float:
        """
        Estimate RMSD using Random Forest regression.
        
        Args:
            features: Array of [clash_score, bond_dev, angle_dev, hydro_score, rg]
            
        Returns:
            Estimated RMSD value
        """
        features = np.array(features).reshape(1, -1)
        rmsd = self.rf_model.predict(features)[0]
        return max(0, rmsd)  # RMSD cannot be negative
    
    def get_per_residue_confidence(
        self,
        coords: np.ndarray,
        sequence: str
    ) -> Tuple[List[float], Dict[str, float], str]:
        """
        Calculate per-residue confidence scores (0-100%).
        
        This is the main QA function that:
        1. Analyzes all structural features
        2. Estimates global RMSD
        3. Computes per-residue confidence scores
        
        Args:
            coords: Backbone atom coordinates (N, CA, C for each residue)
            sequence: Amino acid sequence
            
        Returns:
            confidence_scores: List of 0-100 scores for each residue
            global_metrics: Dictionary of global quality metrics
            quality_report: Human-readable quality report string
        """
        n_residues = len(sequence)
        
        # 1. Detect atomic clashes
        clash_score, clash_pairs = self.detect_atomic_clashes(coords)
        
        # 2. Check bond lengths
        bond_deviation, per_res_bond = self.check_bond_lengths(coords)
        
        # 3. Check bond angles
        angle_deviation, per_res_angle = self.check_bond_angles(coords)
        
        # 4. Calculate hydrophobic packing
        hydro_score, per_res_hydro = self.calculate_hydrophobic_score(coords, sequence)
        
        # 5. Calculate radius of gyration
        rg = self.calculate_radius_of_gyration(coords)
        
        # 6. Estimate global RMSD using Random Forest
        features = [clash_score, bond_deviation, angle_deviation, hydro_score, rg]
        estimated_rmsd = self.estimate_rmsd(features)
        
        # 7. Calculate per-residue confidence scores
        confidence_scores = []
        
        # Find which residues are involved in clashes
        clashing_residues = set()
        for i, j in clash_pairs:
            clashing_residues.add(i)
            clashing_residues.add(j)
        
        for i in range(n_residues):
            # Base confidence from global RMSD (inverted: low RMSD = high confidence)
            base_confidence = max(0, 100 - estimated_rmsd * 8)
            
            # Adjust based on per-residue features
            
            # Penalty for clashes
            if i in clashing_residues:
                base_confidence -= 20
            
            # Penalty for bond length deviation
            if i < len(per_res_bond):
                bond_penalty = min(15, per_res_bond[i] * 20)
                base_confidence -= bond_penalty
            
            # Penalty for angle deviation
            if i < len(per_res_angle):
                angle_penalty = min(10, per_res_angle[i] * 0.3)
                base_confidence -= angle_penalty
            
            # Bonus for good hydrophobic packing
            if i < len(per_res_hydro):
                hydro_bonus = (per_res_hydro[i] - 0.5) * 10
                base_confidence += hydro_bonus
            
            # Terminal residues typically have lower confidence
            if i < 3 or i >= n_residues - 3:
                base_confidence -= 10
            
            # Clamp to 0-100 range
            confidence_scores.append(max(0, min(100, base_confidence)))
        
        # Compile global metrics
        global_metrics = {
            'estimated_rmsd': estimated_rmsd,
            'clash_score': clash_score,
            'num_clashes': len(clash_pairs),
            'avg_bond_deviation': bond_deviation,
            'avg_angle_deviation': angle_deviation,
            'hydrophobic_score': hydro_score,
            'radius_of_gyration': rg,
            'mean_confidence': np.mean(confidence_scores)
        }
        
        # Generate quality report
        quality_report = self._generate_report(global_metrics, confidence_scores, sequence)
        
        return confidence_scores, global_metrics, quality_report
    
    def _generate_report(
        self,
        metrics: Dict[str, float],
        confidences: List[float],
        sequence: str
    ) -> str:
        """Generate a human-readable quality report."""
        
        # Categorize residues by confidence
        high_conf = sum(1 for c in confidences if c >= 70)
        med_conf = sum(1 for c in confidences if 40 <= c < 70)
        low_conf = sum(1 for c in confidences if c < 40)
        
        report = []
        report.append("=" * 50)
        report.append("  QUALITY ASSESSMENT REPORT")
        report.append("=" * 50)
        report.append("")
        report.append(f"Sequence Length: {len(sequence)} residues")
        report.append("")
        report.append("--- Global Metrics ---")
        report.append(f"Estimated RMSD: {metrics['estimated_rmsd']:.2f} Å")
        report.append(f"Mean Confidence: {metrics['mean_confidence']:.1f}%")
        report.append(f"Atomic Clashes: {metrics['num_clashes']}")
        report.append(f"Avg Bond Deviation: {metrics['avg_bond_deviation']:.3f} Å")
        report.append(f"Avg Angle Deviation: {metrics['avg_angle_deviation']:.1f}°")
        report.append(f"Hydrophobic Score: {metrics['hydrophobic_score']:.2f}")
        report.append(f"Radius of Gyration: {metrics['radius_of_gyration']:.1f} Å")
        report.append("")
        report.append("--- Confidence Distribution ---")
        report.append(f"[HIGH]   High (>=70%): {high_conf} residues")
        report.append(f"[MEDIUM] Medium (40-70%): {med_conf} residues")
        report.append(f"[LOW]    Low (<40%): {low_conf} residues")
        report.append("")
        report.append("--- Per-Residue Confidence ---")
        
        # Show confidence in groups of 10
        for i in range(0, len(sequence), 10):
            end = min(i + 10, len(sequence))
            seq_chunk = sequence[i:end]
            conf_chunk = confidences[i:end]
            conf_str = ' '.join([f"{int(c):3d}" for c in conf_chunk])
            report.append(f"{i+1:4d}: {seq_chunk:<10s} | {conf_str}")
        
        report.append("")
        report.append("=" * 50)
        
        return '\n'.join(report)


def get_confidence_color(confidence: float) -> str:
    """
    Get color category based on confidence score.
    
    Args:
        confidence: Score from 0-100
        
    Returns:
        Color category: 'high', 'medium', or 'low'
    """
    if confidence >= 70:
        return 'high'  # Blue
    elif confidence >= 40:
        return 'medium'  # Yellow
    else:
        return 'low'  # Red
