import os
from tqdm import tqdm
from utils.dataUtils import extract_sequences
from dotenv import load_dotenv
import argparse

_ = load_dotenv()


def build_pdb_sequence_database(pdb_dir):
    """Extracts sequences from all PDB files in a directory."""
    pdb_sequences = {}  # Dictionary to store {pdb_id: sequence}

    for pdb_file in tqdm(os.listdir(pdb_dir)):
        if pdb_file.endswith(".pdb"):
            pdb_path = os.path.join(pdb_dir, pdb_file)
            seq = extract_sequences(pdb_path)
            if seq:
                pdb_id = os.path.splitext(pdb_file)[0]  # Extract PDB ID
                pdb_sequences[pdb_id] = seq

    return pdb_sequences  # Dictionary of PDB IDs and sequences


def main():
    parser = argparse.ArgumentParser(description="Build PDB sequence database.")
    parser.add_argument("mode", choices=["train", "test"], help="Mode: train or test")
    args = parser.parse_args()

    if args.mode == "train":
        pdb_directory = os.getenv("TRAIN_PDB_DIR")
        fasta_path = os.getenv("TRAIN_FASTA_PATH")
    elif args.mode == "test":
        pdb_directory = os.getenv("TEST_PDB_DIR")
        fasta_path = os.getenv("TEST_FASTA_PATH")

    pdb_db = build_pdb_sequence_database(pdb_directory)

    # Save extracted sequences to a FASTA file for BLAST
    with open(fasta_path, "w") as fasta_file:
        for pdb_id, seq in pdb_db.items():
            fasta_file.write(f">{pdb_id}\n{seq}\n")

    print(f"Extracted {len(pdb_db)} sequences from PDB database in {args.mode} mode.")


if __name__ == "__main__":
    main()
