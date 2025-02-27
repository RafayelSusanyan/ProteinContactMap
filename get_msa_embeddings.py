import os
import torch
import esm
import numpy as np
from Bio import SeqIO
from tqdm import tqdm
from dotenv import load_dotenv
import argparse

_ = load_dotenv()


def read_main_sequences(fasta_file):
    sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append((record.id, str(record.seq)))
    return sequences


def read_msa_sequences_with_names(fasta_file):
    unique_sequences = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        protein_name = record.id
        sequence = str(record.seq)
        if protein_name not in unique_sequences:
            unique_sequences[protein_name] = sequence
    return list(unique_sequences.items())


def split_msa_sequences(msa_sequences, chunk_size=1023, stride=756):
    seq_length = len(msa_sequences[0][1])
    chunks_list = []
    for start in range(0, seq_length, stride):
        end = start + chunk_size
        if end > seq_length:
            end = seq_length
        chunk_msa = [(seq[0], seq[1][start:end]) for seq in msa_sequences]
        chunks_list.append(chunk_msa)
        if end == seq_length:
            break
    return chunks_list


def combine_embeddings(embeddings_list, overlap_stride=268):
    # Ensure embeddings_list is not empty
    if not embeddings_list:
        raise ValueError("embeddings_list cannot be empty")

    # Initialize the combined embedding with the first embedding in the list
    combined_embedding = torch.from_numpy(embeddings_list[0]).float()
    combined_length = combined_embedding.shape[2]

    # Iterate over the rest of the embeddings
    for embedding in embeddings_list[1:]:
        current_length = embedding.shape[2]

        # Convert the current embedding to a PyTorch tensor
        embedding_tensor = torch.from_numpy(embedding).float()

        # Calculate the overlap region
        overlap_start_combined = combined_length - overlap_stride
        overlap_end_combined = combined_length
        overlap_start_current = 0
        overlap_end_current = overlap_stride

        # Compute the mean in the overlap region
        combined_embedding[:, :, overlap_start_combined:overlap_end_combined, :] = (
                                                                                           combined_embedding[:, :,
                                                                                           overlap_start_combined:overlap_end_combined,
                                                                                           :] +
                                                                                           embedding_tensor[:, :,
                                                                                           overlap_start_current:overlap_end_current,
                                                                                           :]
                                                                                   ) / 2

        # Concatenate the non-overlapping part
        combined_embedding = torch.cat((combined_embedding, embedding_tensor[:, :, overlap_end_current:, :]), dim=2)
        combined_length = combined_embedding.shape[2]

    # Convert the combined embedding back to a NumPy array
    combined_embedding = combined_embedding.numpy()[:, :, :-1, :]
    return combined_embedding


def get_msa_embeddings(msa_model, msa_alphabet, msa_file, save_path, device="cuda"):
    # Read MSA sequences
    emb_list = []
    msa_data = read_msa_sequences_with_names(msa_file)
    # msa_data = get_unique_sequences(msa_data)
    chunked = split_msa_sequences(msa_data)
    for chunk_msa in chunked:
        batch_converter = msa_alphabet.get_batch_converter()
        batch_labels, batch_strs, batch_tokens = batch_converter(chunk_msa)
        batch_tokens = batch_tokens.to(device)

        # Run model
        with torch.no_grad():
            results = msa_model(batch_tokens, repr_layers=[12], return_contacts=False)
        msa_embeddings = results["representations"][12].cpu().numpy()
        emb_list.append(msa_embeddings)

    emb_final = combine_embeddings(emb_list)
    # Save embeddings
    np.savez(save_path, msa_embeddings=emb_final)


# Load pretrained model
msa_model, msa_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
device = "cuda" if torch.cuda.is_available() else "cpu"
msa_model = msa_model.to(device)
msa_model.eval()  # Set the model to evaluation mode

def main():
    parser = argparse.ArgumentParser(description="Process PDB sequences and compute aligned MSA.")
    parser.add_argument("mode", choices=["train", "test"], help="Mode: train or test")
    args = parser.parse_args()

    if args.mode == "train":
        main_sequences_file = os.getenv("TRAIN_FASTA_PATH")
        embeddings_dir = os.getenv("TRAIN_MSA_STRINGS_DIR")
        save_dir = os.getenv("TRAIN_MSA_EMBEDDING_DIR")
    elif args.mode == "test":
        main_sequences_file = os.getenv("TEST_FASTA_PATH")
        embeddings_dir = os.getenv("TEST_MSA_STRINGS_DIR")
        save_dir = os.getenv("TEST_MSA_EMBEDDING_DIR")

    os.makedirs(save_dir, exist_ok=True)

    main_sequences = read_main_sequences(main_sequences_file)

    for protein_id, sequence in tqdm(main_sequences):
        # Path to aligned MSA file for the current sequence
        msa_file = os.path.join(embeddings_dir, protein_id, "aligned_msa.fasta")

        # Save path for embeddings
        save_path = os.path.join(save_dir, f"{protein_id}_msa_embeddings.npz")

        # Check if embeddings already exist
        if os.path.exists(save_path):
            print(f"Embeddings for {protein_id} already exist. Skipping.")
            continue

        # Generate and save MSA embeddings
        try:
            get_msa_embeddings(msa_model, msa_alphabet, msa_file, save_path, device=device)
        except Exception as e:
            print(f"Error processing {protein_id}: {e}")
            continue

    print("All embeddings have been generated and saved.")


if __name__ == "__main__":
    main()

