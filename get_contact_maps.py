import argparse
import os
import numpy as np
from Bio import SeqIO
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from utils.dataUtils import extract_sequence_and_contact_map
from dotenv import load_dotenv

_ = load_dotenv()

fasta_file = None
pdb_dir = None
contact_map_dir = None
distance_map_dir = None
fasta_sequences = None


def process_pdb(pdb_filename):
    """Processes a single PDB file, extracting sequence and computing contact map."""
    pdb_id = os.path.splitext(pdb_filename)[0]  # Extract file ID without extension
    pdb_file = os.path.join(pdb_dir, pdb_filename)
    contact_map_file = os.path.join(contact_map_dir, f"contact_map_{pdb_id}.npz")
    distance_map_file = os.path.join(distance_map_dir, f"contact_map_{pdb_id}.npz")

    # **Skip processing if the contact map already exists**
    if os.path.exists(distance_map_file):
        return f"✅ Skipping {pdb_filename}, contact map already exists."

    # **Check if the PDB ID exists in the FASTA file**
    if pdb_id not in fasta_sequences:
        return f"⚠ Warning: No matching FASTA sequence for {pdb_id}. Skipping."

    fasta_sequence = fasta_sequences[pdb_id]

    try:
        sequence_str, dist_matrix, contact_map = extract_sequence_and_contact_map(pdb_file)

        # **Check if sequence length matches FASTA**
        if len(sequence_str) != len(fasta_sequence):
            return f"❌ Mismatch: FASTA length ({len(fasta_sequence)}) ≠ PDB extracted length ({len(sequence_str)}) for {pdb_id}"

        # Save Contact Map and Distance Map
        np.savez(contact_map_file, contact_map=contact_map)
        np.savez(distance_map_file, contact_map=dist_matrix)

        return f"✅ Processed {pdb_filename}, saved contact map."

    except Exception as e:
        return f"❌ Error processing {pdb_filename}: {e}"


def main():
    parser = argparse.ArgumentParser(description="Process PDB files and extract contact maps.")
    parser.add_argument("mode", choices=["train", "test"], help="Mode: train or test")
    args = parser.parse_args()

    global fasta_file, pdb_dir, contact_map_dir, distance_map_dir, fasta_sequences

    if args.mode == "train":
        fasta_file = os.getenv("TRAIN_FASTA_PATH")
        pdb_dir = os.getenv("TRAIN_PDB_DIR")
        contact_map_dir = os.getenv("TRAIN_CONTACT_MAP_DIR")
        distance_map_dir = os.getenv("TRAIN_DISTANCE_MAP_DIR")
        fasta_sequences = {record.id: str(record.seq) for record in SeqIO.parse(fasta_file, "fasta")}
    elif args.mode == "test":
        fasta_file = os.getenv("TEST_FASTA_PATH")
        pdb_dir = os.getenv("TEST_PDB_DIR")
        contact_map_dir = os.getenv("TEST_CONTACT_MAP_DIR")
        distance_map_dir = os.getenv("TEST_DISTANCE_MAP_DIR")
        fasta_sequences = {record.id: str(record.seq) for record in SeqIO.parse(fasta_file, "fasta")}

    os.makedirs(contact_map_dir, exist_ok=True)
    os.makedirs(distance_map_dir, exist_ok=True)

    # Get all PDB files in the directory
    pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith(".pdb")]

    # Run Parallel Processing
    num_workers = min(cpu_count(), len(pdb_files))  # Use max available CPUs
    print(f"🚀 Running with {num_workers} parallel workers...")

    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(process_pdb, pdb_files), total=len(pdb_files)))

    # Print final status
    for res in results:
        print(res)

    print(f"✅ Processing complete! Contact maps saved in: {contact_map_dir}")
    print(f"✅ Processing complete! Distance maps saved in: {distance_map_dir}")


if __name__ == "__main__":
    main()
