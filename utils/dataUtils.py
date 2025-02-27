import numpy as np
from Bio import PDB
from Bio.SeqUtils import seq1

VALID_RESIDUES = {
    "ALA": "A", "ARG": "R", "ASP": "D", "CYS": "C", "CYX": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "HIE": "H",
    "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "ASN": "N",
    "PHE": "F", "PRO": "P", "SEC": "U", "SER": "S", "THR": "T",
    "TRP": "W", "TYR": "Y", "VAL": "V"
}

def extract_sequences(pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("molecule", pdb_file)

    sequences = {}

    for model in structure:
        for chain in model:
            chain_id = chain.id
            residues = [res for res in chain if res.id[0] == " "]  # Ignore heteroatoms

            sequence = "".join(seq1(res.resname) for res in residues if res.resname in VALID_RESIDUES.keys())
            if sequence:
                sequences[chain_id] = sequence  # Store sequence for each chain

    arr = []
    for k, v in sequences.items():
      arr.append(v)

    return "".join(arr)


def extract_sequence_and_contact_map(pdb_file, threshold=8.0):
    """
    Extracts protein sequence and saves Cα coordinates from ALL chains in a PDB file.
    Note that it ignores HETATM
    """
    parser = PDB.PDBParser(QUIET=True)
    # structure = parser.get_structure("protein", pdb_file)
    structure = parser.get_structure("molecule", pdb_file)

    sequence = {}
    seq = []
    ca_coord = []
    ca_coordinates = {}

    for model in structure:
        for chain in model:
            residues = [res for res in chain if res.id[0] == " "]  # Ignoring HETATM
            for residue in residues:
                resname = residue.resname  # Get 3-letter residue name

                if resname in VALID_RESIDUES:  # Keep only allowed residues
                    seq.append(VALID_RESIDUES[resname])  # Convert to 1-letter code

                    # Get Cα atom coordinates
                    if "CA" in residue:
                        ca_coord.append(residue["CA"].coord)
                    else:
                        ca_coord.append(np.array([np.nan, np.nan, np.nan]))  # Placeholder for missing Cα
            sequence[chain.id] = seq
            ca_coordinates[chain.id] = ca_coord
            seq = []
            ca_coord = []

    arr = []
    arr2 = []
    for k, v in sequence.items():
        arr.append("".join(v))
        arr2.append(ca_coordinates[k])
    sequence_str = "".join(arr)
    ca_coordinates = np.vstack(arr2)  # Convert to NumPy array
    # Compute distance map
    dist_matrix = np.linalg.norm(ca_coordinates[:, np.newaxis, :] - ca_coordinates[np.newaxis, :, :], axis=-1)
    contact_map = (dist_matrix < threshold).astype(int)
    return sequence_str, dist_matrix, contact_map
