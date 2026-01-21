import numpy as np

def place_atom(a, b, c, bond_len, bond_angle, torsion_angle):
    """
    NeRF (Natural Extension of Reference Frame) algorithm to place the next atom
    based on the previous 3 atoms (a, b, c) and internal coordinates.
    
    Args:
        a, b, c: Previous three atom positions
        bond_len: Distance from c to new atom
        bond_angle: Angle at c (b-c-new) in radians
        torsion_angle: Dihedral angle (a-b-c-new) in radians
    """
    # Vector from b to c
    bc = c - b
    bc_norm = bc / np.linalg.norm(bc)
    
    # Vector from a to b
    ab = b - a
    
    # Normal to plane abc
    n = np.cross(ab, bc)
    n_len = np.linalg.norm(n)
    if n_len < 1e-10:
        # Fallback for collinear atoms
        n = np.array([0.0, 0.0, 1.0])
    else:
        n = n / n_len
    
    # Vector in plane, perpendicular to bc
    m = np.cross(n, bc_norm)
    
    # Build new atom position using internal coordinates
    d_x = bond_len * np.cos(np.pi - bond_angle)
    d_y = bond_len * np.sin(np.pi - bond_angle) * np.cos(torsion_angle)
    d_z = bond_len * np.sin(np.pi - bond_angle) * np.sin(torsion_angle)
    
    new_pos = c + d_x * bc_norm + d_y * m + d_z * n
    return new_pos

def angles_to_coordinates(phi_psi_angles):
    """
    Converts a list of (Phi, Psi) angles into a backbone (N, CA, C) 3D trace.
    
    Uses standard protein geometry:
    - Bond lengths: N-CA = 1.458Å, CA-C = 1.525Å, C-N = 1.329Å
    - Bond angles: ~111° (tau) at CA, ~116° at C, ~121° at N
    - Omega (peptide bond) fixed at 180° (trans)
    """
    coords = []
    
    # Standard bond lengths (in Angstroms)
    BOND_N_CA = 1.458
    BOND_CA_C = 1.525
    BOND_C_N = 1.329
    
    # Standard bond angles (in radians)
    ANGLE_N_CA_C = np.radians(111.2)   # tau angle at CA
    ANGLE_CA_C_N = np.radians(116.2)   # angle at C
    ANGLE_C_N_CA = np.radians(121.7)   # angle at N
    
    # Omega angle (peptide bond - always trans = 180°)
    OMEGA = np.pi
    
    # Initialize first residue with proper geometry
    n0 = np.array([0.0, 0.0, 0.0])
    ca0 = np.array([BOND_N_CA, 0.0, 0.0])
    c0 = np.array([BOND_N_CA + BOND_CA_C * np.cos(np.pi - ANGLE_N_CA_C),
                   BOND_CA_C * np.sin(np.pi - ANGLE_N_CA_C), 
                   0.0])
    
    coords.extend([n0, ca0, c0])
    
    # Build the chain using proper backbone geometry
    for i, (phi, psi) in enumerate(phi_psi_angles):
        # Get last 3 atoms (CA, C of previous residue and will add N, CA, C of next)
        ca_prev = coords[-2]
        c_prev = coords[-1]
        n_prev = coords[-3]
        
        # 1. Place next Nitrogen using psi angle (C-N bond, angle at C, psi torsion)
        n_next = place_atom(n_prev, ca_prev, c_prev, BOND_C_N, ANGLE_CA_C_N, psi)
        coords.append(n_next)
        
        # 2. Place next CA using omega angle (fixed at 180° for trans peptide)
        ca_next = place_atom(ca_prev, c_prev, n_next, BOND_N_CA, ANGLE_C_N_CA, OMEGA)
        coords.append(ca_next)
        
        # 3. Place next C using phi angle
        c_next = place_atom(c_prev, n_next, ca_next, BOND_CA_C, ANGLE_N_CA_C, phi)
        coords.append(c_next)
    
    return np.array(coords)


def distance_matrix_to_coordinates(distance_matrix, n_dims=3):
    """
    Convert a distance matrix to 3D coordinates using classical MDS
    (Multidimensional Scaling) with chain connectivity enforcement.
    
    This is the algorithm used to reconstruct Cα coordinates from 
    predicted pairwise distances.
    
    Args:
        distance_matrix: (L × L) symmetric matrix of pairwise distances
        n_dims: Number of dimensions for output (default 3 for 3D)
        
    Returns:
        coordinates: (L × 3) array of 3D coordinates
    """
    n = distance_matrix.shape[0]
    
    # Make symmetric (average upper and lower triangular)
    D = (distance_matrix + distance_matrix.T) / 2
    np.fill_diagonal(D, 0)
    
    # CRITICAL: Enforce chain connectivity - sequential Cα atoms must be ~3.8Å apart
    CA_CA_DISTANCE = 3.8  # Standard Cα-Cα distance in Angstroms
    for i in range(n - 1):
        D[i, i+1] = CA_CA_DISTANCE
        D[i+1, i] = CA_CA_DISTANCE
    
    # Step 1: Compute squared distance matrix
    D_sq = D ** 2
    
    # Step 2: Double centering to get Gram matrix (inner product matrix)
    # B = -0.5 * J * D² * J, where J = I - (1/n) * 1 * 1^T
    row_mean = np.mean(D_sq, axis=1, keepdims=True)
    col_mean = np.mean(D_sq, axis=0, keepdims=True)
    total_mean = np.mean(D_sq)
    
    B = -0.5 * (D_sq - row_mean - col_mean + total_mean)
    
    # Step 3: Eigendecomposition of Gram matrix
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    
    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Step 4: Take top n_dims eigenvectors
    # Clamp negative eigenvalues to small positive (numerical stability)
    eigenvalues = np.maximum(eigenvalues[:n_dims], 1e-10)
    eigenvectors = eigenvectors[:, :n_dims]
    
    # Step 5: Compute coordinates: X = V * sqrt(Lambda)
    coordinates = eigenvectors * np.sqrt(eigenvalues)
    
    return coordinates


def distance_to_full_backbone(ca_coords, sequence):
    """
    Expand Cα-only coordinates to full backbone (N, CA, C) atoms.
    Uses the actual Cα positions and places N/C atoms using proper geometry.
    
    This ensures a connected backbone chain.
    
    Args:
        ca_coords: (L × 3) array of Cα coordinates
        sequence: Amino acid sequence (for length verification)
        
    Returns:
        backbone_coords: (L*3 × 3) array with N, CA, C for each residue
    """
    n_residues = len(ca_coords)
    backbone = []
    
    # Standard bond lengths and geometry
    BOND_N_CA = 1.458  # Å
    BOND_CA_C = 1.525  # Å
    
    for i in range(n_residues):
        ca = ca_coords[i]
        
        # Calculate local coordinate frame from neighboring Cα atoms
        if i == 0:
            # First residue
            if n_residues > 1:
                forward = ca_coords[1] - ca
            else:
                forward = np.array([1.0, 0.0, 0.0])
        elif i == n_residues - 1:
            # Last residue
            forward = ca - ca_coords[i-1]
        else:
            # Middle residues - use vector from prev to next
            forward = ca_coords[i+1] - ca_coords[i-1]
        
        # Normalize forward vector
        forward_len = np.linalg.norm(forward)
        if forward_len < 0.01:
            forward = np.array([1.0, 0.0, 0.0])
        else:
            forward = forward / forward_len
        
        # Create orthogonal vectors for local frame
        # Use a reference vector not parallel to forward
        if abs(forward[0]) < 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        else:
            ref = np.array([0.0, 1.0, 0.0])
        
        # Create perpendicular vectors
        right = np.cross(forward, ref)
        right = right / (np.linalg.norm(right) + 1e-10)
        up = np.cross(right, forward)
        up = up / (np.linalg.norm(up) + 1e-10)
        
        # Place N atom (behind and above CA)
        n_pos = ca - forward * (BOND_N_CA * 0.7) + up * (BOND_N_CA * 0.7)
        
        # CA at its predicted position
        ca_pos = ca
        
        # Place C atom (ahead and below CA)
        c_pos = ca + forward * (BOND_CA_C * 0.7) - up * (BOND_CA_C * 0.7)
        
        backbone.extend([n_pos, ca_pos, c_pos])
    
    return np.array(backbone)