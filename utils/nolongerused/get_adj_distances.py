# preprocess_average_embeddings.py

import os
import numpy as np
from Bio import SeqIO, AlignIO
from multiprocessing import Pool, cpu_count
from utils.blastUtils import make_blast_db, parse_blast_results, run_blast


def get_alignment_indices(aligned_target_seq, aligned_similar_seq):
    target_indices = []
    similar_indices = []
    idx_target = 0
    idx_similar = 0
    for res_target, res_similar in zip(aligned_target_seq, aligned_similar_seq):
        if res_target != '-' and res_similar != '-':
            target_indices.append(idx_target)
            similar_indices.append(idx_similar)
        if res_target != '-':
            idx_target += 1
        if res_similar != '-':
            idx_similar += 1
    return target_indices, similar_indices


def preprocess_single_sequence(args):
    protein_id, sequence, alignment_dict, pdb_dir, distance_map_dir, blast_db_name, blast_dir, max_similar_sequences, adjusted_distance_map_dir = args

    num_nodes = len(sequence)

    # Paths
    blast_output_file = os.path.join(blast_dir, f'blast_results_{protein_id}.txt')
    adjusted_distance_map_file = os.path.join(adjusted_distance_map_dir, f"adjusted_distance_map_{protein_id}.npz")

    if os.path.exists(adjusted_distance_map_file):
        # Skipping processing for this protein
        return protein_id, False  # Return protein_id and False indicating it was skipped

    run_blast(sequence, blast_db_name, blast_output_file, protein_id, blast_dir, max_target_seqs=max_similar_sequences)
    similar_sequences = parse_blast_results(blast_output_file)
    similar_sequences = [seq_id for seq_id in similar_sequences if seq_id != protein_id]

    structural_features = []

    aligned_target_seq = alignment_dict.get(protein_id)
    if not aligned_target_seq:
        print(f"Aligned sequence for {protein_id} not found in alignment dictionary.")
        return

    for similar_seq_id in similar_sequences:
        aligned_similar_seq = alignment_dict.get(similar_seq_id)
        if not aligned_similar_seq:
            continue

        target_indices, similar_indices = get_alignment_indices(aligned_target_seq, aligned_similar_seq)

        # Load distance map of similar sequence
        distance_map_file = os.path.join(distance_map_dir, f"contact_map_{similar_seq_id}.npz")
        if os.path.exists(distance_map_file):
            distance_map = np.load(distance_map_file)["contact_map"]
            # Adjust distance map based on alignment
            adjusted_distance_map = np.full((num_nodes, num_nodes), np.inf)
            for i, idx_i in enumerate(similar_indices):
                for j, idx_j in enumerate(similar_indices):
                    if idx_i < distance_map.shape[0] and idx_j < distance_map.shape[1]:
                        adjusted_distance_map[target_indices[i], target_indices[j]] = distance_map[idx_i, idx_j]
            structural_features.append(adjusted_distance_map)
        else:
            print(f"Distance map for {similar_seq_id} not found.")

    if structural_features:
        avg_distance_map = np.mean(structural_features, axis=0)
        # Save the average adjusted distance map
        np.savez_compressed(adjusted_distance_map_file, adjusted_distance_map=avg_distance_map)
        print(f"Saved adjusted distance map for {protein_id}.")
    else:
        # If no structural features, save zeros
        zeros_map = np.zeros((num_nodes, num_nodes))
        np.savez_compressed(adjusted_distance_map_file, adjusted_distance_map=zeros_map)
        print(f"No structural features found for {protein_id}. Saved zeros.")


def main():
    # Paths to your data directories (update these paths accordingly)
    fasta_file = os.getenv("TRAIN_FASTA_PATH")
    a3m_file = os.getenv('TRAIN_A3M_PATH')
    pdb_dir = os.getenv('data/train/pdb/')
    distance_map_dir = os.getenv('TRAIN_DISTANCE_MAP_DIR')
    blast_db_name = os.getenv('BLAST_DB_NAME')
    blast_dir = os.getenv('BLAST_DIR')        # Directory to store BLAST results
    max_similar_sequences = 5                   # Maximum number of similar sequences to consider
    adjusted_distance_map_dir = os.getenv('TRAIN_ADJUSTED_DISTANCE_MAP_DIR')

    # Ensure output directories exist
    os.makedirs(blast_dir, exist_ok=True)
    os.makedirs(adjusted_distance_map_dir, exist_ok=True)

    # Create BLAST database if it doesn't exist
    if not os.path.exists(blast_db_name + '.pin'):
        make_blast_db(fasta_file, blast_db_name)

    # Load sequences and alignment
    sequences = {}
    for record in SeqIO.parse(fasta_file, 'fasta'):
        sequences[record.id] = str(record.seq)
    alignment = AlignIO.read(a3m_file, 'fasta')
    alignment_dict = {record.id: str(record.seq) for record in alignment}

    # Prepare arguments for multiprocessing
    args_list = []
    for protein_id, sequence in sequences.items():
        args = (protein_id, sequence, alignment_dict, pdb_dir, distance_map_dir, blast_db_name, blast_dir, max_similar_sequences, adjusted_distance_map_dir)
        args_list.append(args)

    num_workers = min(cpu_count(), len(args_list))  # Use max available CPUs
    print(f"🚀 Running with {num_workers} parallel workers...")

    with Pool(processes=num_workers) as pool:
        pool.map(preprocess_single_sequence, args_list)

if __name__ == '__main__':
    main()
