def save_pdb(coords, sequence, confidence_scores, filename="output/prediction.pdb"):
    """
    Writes coordinates to a PDB file.
    CRITICAL: Maps 'confidence_scores' to the B-Factor column for color coding.
    """
    # 3 atoms per residue (N, CA, C)
    # Ensure coords match sequence length * 3
    
    with open(filename, 'w') as pdb:
        atom_idx = 1
        res_idx = 1
        
        for i, aa in enumerate(sequence):
            # We have 3 backbone atoms per residue in our simplified model
            residue_atoms = ["N", "CA", "C"]
            
            # Get the confidence score for this residue
            # If we only have a global score, use it for all. 
            # If we have local scores, use the specific one.
            if isinstance(confidence_scores, list):
                b_factor = confidence_scores[i]
            else:
                b_factor = confidence_scores # Global score applied to all
            
            for j, atom_type in enumerate(residue_atoms):
                # Calculate index in the coords array
                # (Skipping the initialization atoms for simplicity in this demo logic)
                coord_idx = (i * 3) + j 
                if coord_idx >= len(coords): break
                
                x, y, z = coords[coord_idx]
                
                # PDB ATOM Format Specs:
                # Columns 61-66 is the Temperature Factor (B-Factor). 
                # We overwrite this with our Confidence Score.
                # Format: "ATOM  {ID}  {Type}  {Res} {Chain} {ResID}    {X}   {Y}   {Z}  {Occ} {B-Factor}"
                
                line = "{:6s}{:5d} {:^4s} {:3s} {:1s}{:4d}    {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}\n".format(
                    "ATOM", atom_idx, atom_type, aa, "A", res_idx, x, y, z, 1.00, b_factor
                )
                pdb.write(line)
                atom_idx += 1
            
            res_idx += 1
        
        print(f"[OK] PDB Saved to: {filename}")
        print(f"[i]  To view color coding: Open in PyMOL -> 'Spectrum b-factors'")