import os
import argparse
import subprocess
import numpy as np
from Bio import SeqIO
from tqdm import tqdm
from dotenv import load_dotenv
from utils.blastUtils import make_blast_db

_ = load_dotenv()

# Set base output directory
BASE_OUTPUT_DIR = None
BLAST_DB = os.getenv('BLAST_DB_NAME')

# Step 1: Run PSI-BLAST
def run_psiblast(input_fasta, output_dir):
    """Runs PSI-BLAST to find homologous sequences."""
    psi_blast_output = os.path.join(output_dir, "psiblast_output.xml")
    # Another way to do it
    # command = (
    #     f"psiblast -query {input_fasta} -db {BLAST_DB} -num_iterations 3 "
    #     f"-evalue 0.001 -max_target_seqs 20 -inclusion_ethresh 1e-6 "
    #     f"-outfmt 5 -out {psi_blast_output}"
    # )
    command = (
        f"psiblast -query {input_fasta} -db {BLAST_DB} -num_iterations 3 "
        f"-max_target_seqs 5 "
        f"-outfmt 5 -out {psi_blast_output}"
    )
    subprocess.run(command, shell=True, check=True)
    return psi_blast_output

def remove_duplicates_msa(input_fasta, output_fasta):
    unique_sequences = set()
    unique_records = []

    # Read the input FASTA file and identify unique sequences
    for record in SeqIO.parse(input_fasta, "fasta"):
        sequence = str(record.seq)
        if sequence not in unique_sequences:
            unique_sequences.add(sequence)
            unique_records.append(record)

    with open(output_fasta, "w") as output_handle:
        SeqIO.write(unique_records, output_handle, "fasta")

def contains_single_fasta(fasta_file):
    count = 0
    for _ in SeqIO.parse(fasta_file, "fasta"):
        count += 1
        if count > 1:
            return False
    return count == 1

# Step 2: Extract Homologous Sequences
def extract_msa_from_psiblast(input_fasta, psi_blast_output, output_dir):
    """Extracts homologous sequences from PSI-BLAST output."""
    sequence_ids_output = os.path.join(output_dir, "sequence_ids.txt")
    msa_fasta = os.path.join(output_dir, "msa.fasta")

    command = f"grep -oP '(?<=<Hit_id>).*?(?=</Hit_id>)' {psi_blast_output} > {sequence_ids_output}"
    subprocess.run(command, shell=True, check=True)

    command2 = (
        f'blastdbcmd -db {BLAST_DB} -entry_batch {sequence_ids_output} '
        f'-outfmt "%f" > {msa_fasta}'
    )
    tmp_msa = msa_fasta.split('.')[0] + "_tmp.fasta"
    subprocess.run(command2, shell=True, check=True)
    command3 = f"cat {input_fasta} {msa_fasta} > {tmp_msa}"
    subprocess.run(command3, shell=True, check=True)

    remove_duplicates_msa(tmp_msa, tmp_msa)
    return msa_fasta

# Step 3: Align MSA Using Clustal Omega
def align_msa(msa_fasta, output_dir):
    """Aligns the MSA using Clustal Omega."""
    aligned_msa = os.path.join(output_dir, "aligned_msa.fasta")
    tmp_msa = msa_fasta.split('.')[0] + "_tmp.fasta"
    if contains_single_fasta(tmp_msa):
        command_cpy = f"cp {tmp_msa} {aligned_msa}"
        subprocess.run(command_cpy, shell=True, check=True)
        return aligned_msa
    command = f"clustalo -i {tmp_msa} -o {aligned_msa} --force"
    subprocess.run(command, shell=True, check=True)
    return aligned_msa

# Main Function to Process All Sequences in a FASTA File
def process_fasta_file(fasta_file, output_dir):
    """Processes each sequence in a large FASTA file."""
    records = list(SeqIO.parse(fasta_file, "fasta"))
    total_sequences = len(records)
    os.makedirs(output_dir, exist_ok=True)

    # Using tqdm to create a progress bar
    for record in tqdm(records, total=total_sequences, desc="Processing Sequences"):
        sequence_id = record.id
        # Create directory for this sequence
        sequence_output_dir = os.path.join(output_dir, sequence_id)
        os.makedirs(sequence_output_dir, exist_ok=True)

        # Save sequence as individual FASTA file
        individual_fasta = os.path.join(sequence_output_dir, f"{sequence_id}.fasta")
        with open(individual_fasta, "w") as f:
            f.write(f">{sequence_id}\n{record.seq}\n")
        try:
            psi_blast_output = run_psiblast(individual_fasta, sequence_output_dir)
            msa_fasta = extract_msa_from_psiblast(individual_fasta, psi_blast_output, sequence_output_dir)
            aligned_msa = align_msa(msa_fasta, sequence_output_dir)
        except Exception as e:
            print(f"Error processing {sequence_id}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Process PDB sequences and compute aligned MSA.")
    parser.add_argument("mode", choices=["train", "test"], help="Mode: train or test")
    args = parser.parse_args()

    if args.mode == "train":
        fasta_file = os.getenv("TRAIN_FASTA_PATH")
    elif args.mode == "test":
        fasta_file = os.getenv("TEST_FASTA_PATH")

    BASE_OUTPUT_DIR = os.getenv(f"{str(args.mode).upper()}_MSA_STRINGS_DIR")
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    make_blast_db(fasta_file=fasta_file, db_name=BLAST_DB)
    process_fasta_file(fasta_file, BASE_OUTPUT_DIR)

if __name__ == "__main__":
    main()