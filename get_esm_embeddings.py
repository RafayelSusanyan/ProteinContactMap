import os
import argparse
import torch
import numpy as np
from Bio import SeqIO
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from dotenv import load_dotenv

_ = load_dotenv()

# Enable GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# Load ESM-2 Model
MODEL_NAME = os.getenv("EMBEDDING_MODEL")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()


def get_esm2_batch_embeddings(sequences):
    """Compute ESM-2 embeddings for a batch of sequences."""
    tokens = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False)
    tokens = {key: val.to(device) for key, val in tokens.items()}  # Move to GPU
    with torch.no_grad():
        output = model(**tokens).last_hidden_state
    return output.cpu().numpy()  # Move back to CPU for saving


def main():
    parser = argparse.ArgumentParser(description="Process PDB sequences and compute embeddings.")
    parser.add_argument("mode", choices=["train", "test"], help="Mode: train or test")
    args = parser.parse_args()

    if args.mode == "train":
        fasta_file = os.getenv("TRAIN_FASTA_PATH")
        output_dir = os.getenv("TRAIN_EMBEDDING_DIR")
    elif args.mode == "test":
        fasta_file = os.getenv("TEST_FASTA_PATH")
        output_dir = os.getenv("TEST_EMBEDDING_DIR")

    os.makedirs(output_dir, exist_ok=True)

    # Load sequences
    batch_size = 1  # Increase batch size for GPUs
    sequences = []
    sequence_ids = []

    for record in tqdm(SeqIO.parse(fasta_file, "fasta"), desc="Loading sequences"):
        sequence_id = record.id
        embedding_file = os.path.join(output_dir, f"embedding_{sequence_id}.npz")

        # Skip processing if the file already exists
        if os.path.exists(embedding_file):
            print(f"✅ Skipping {sequence_id}, embedding already exists.")
            continue

        sequence_ids.append(sequence_id)
        sequences.append(str(record.seq))

    # Process sequences in batches
    for i in tqdm(range(0, len(sequences), batch_size), desc="Processing sequences in batches"):
        batch_sequences = sequences[i:i + batch_size]
        batch_ids = sequence_ids[i:i + batch_size]

        # Compute batch embeddings
        batch_embeddings = get_esm2_batch_embeddings(batch_sequences)

        # Save each embedding separately
        for idx, embedding in enumerate(batch_embeddings):
            embedding_file = os.path.join(output_dir, f"embedding_{batch_ids[idx]}.npz")
            np.savez(embedding_file, embedding=embedding)
        torch.cuda.empty_cache()

    print(f"✅ Processing complete! Embeddings saved in: {output_dir}")


if __name__ == "__main__":
    main()
