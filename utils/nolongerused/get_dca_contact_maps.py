import os
import subprocess
import numpy as np
from Bio import SeqIO
from utils.blastUtils import make_blast_db

# Set base output directory
BASE_OUTPUT_DIR = "data/train/dca_contact_maps"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# Define paths for tools
BLAST_DB = "./blast/protein_db"  # Change this to your BLAST database path
GREMLIN_DIR = "GREMLIN_CPP"


# Step 1: Run PSI-BLAST
def run_psiblast(input_fasta, output_dir):
    """Runs PSI-BLAST to find homologous sequences."""
    psi_blast_output = os.path.join(output_dir, "psiblast_output.xml")
    command = f"psiblast -query {input_fasta} -db {BLAST_DB} -num_iterations 3 -evalue 0.001 -max_target_seqs 100 -inclusion_ethresh 1e-6 -outfmt 5 -out {psi_blast_output}"
    subprocess.run(command, shell=True, check=True)
    print(f"PSI-BLAST completed for {input_fasta}")
    return psi_blast_output


# Step 2: Extract Homologous Sequences
def extract_msa_from_psiblast(psi_blast_output, output_dir):
    """Extracts homologous sequences from PSI-BLAST output."""
    sequence_ids_output = os.path.join(output_dir, "sequence_ids.txt")
    msa_fasta = os.path.join(output_dir, "msa.fasta")

    command = f"grep -oP '(?<=<Hit_id>).*?(?=</Hit_id>)' {psi_blast_output} > {sequence_ids_output}"
    subprocess.run(command, shell=True, check=True)

    command2 = f'blastdbcmd -db {BLAST_DB} -entry_batch {sequence_ids_output} -outfmt "%f" > {msa_fasta}'
    subprocess.run(command2, shell=True, check=True)

    print(f"MSA saved to {msa_fasta}")
    return msa_fasta


# Step 3: Align MSA Using Clustal Omega
def align_msa(msa_fasta, output_dir):
    """Aligns the MSA using Clustal Omega."""
    aligned_msa = os.path.join(output_dir, "aligned_msa.fasta")
    command = f"clustalo -i {msa_fasta} -o {aligned_msa} --force"
    subprocess.run(command, shell=True, check=True)
    print(f"Aligned MSA saved to {aligned_msa}")
    return aligned_msa


# Step 4: Run GREMLIN
def run_gremlin(aligned_msa, output_dir):
    """Runs GREMLIN for DCA on the aligned MSA."""
    gremlin_output = os.path.join(output_dir, "gremlin_output.txt")
    command = f"./{GREMLIN_DIR}/gremlin_cpp -i {aligned_msa} -o {gremlin_output}"
    subprocess.run(command, shell=True, check=True)
    print(f"GREMLIN output saved to {gremlin_output}")
    return gremlin_output


# Step 5: Parse GREMLIN Output
def parse_gremlin_output(file_path):
    """Parses GREMLIN output to extract DCA scores."""
    dca_results = []
    with open(file_path, 'r') as file:
        for idx, line in enumerate(file):
            if idx == 0:  # Skip header
                continue
            parts = line.strip().split()
            i, j = int(parts[0]), int(parts[1])
            raw_score, apc_score = float(parts[2]), float(parts[3])
            ii, jj = parts[4], parts[5]  # Ignored for now
            dca_results.append((i, j, raw_score, apc_score, ii, jj))
    return dca_results


# Step 6: Create Contact Map
def create_contact_map(dca_results, sequence_length):
    """Creates a contact map from DCA results."""
    contact_map = np.zeros((sequence_length, sequence_length))
    for i, j, raw_score, apc_score, ii, jj in dca_results:
        contact_map[i - 1, j - 1] = apc_score
        contact_map[j - 1, i - 1] = apc_score  # Symmetric matrix
    return contact_map


# Step 7: Save Contact Map
def save_contact_map(contact_map, output_dir):
    """Saves contact map as numpy array."""
    contact_map_file = os.path.join(output_dir, "contact_map.npy")
    np.save(contact_map_file, contact_map)
    print(f"Contact map saved as {contact_map_file}")


# Main Function to Process All Sequences in a FASTA File
def process_fasta_file(fasta_file):
    """Processes each sequence in a large FASTA file."""
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequence_id = record.id
        sequence_length = len(record.seq)

        # Create directory for this sequence
        sequence_output_dir = os.path.join(BASE_OUTPUT_DIR, sequence_id)
        os.makedirs(sequence_output_dir, exist_ok=True)

        # Save sequence as individual FASTA file
        individual_fasta = os.path.join(sequence_output_dir, f"{sequence_id}.fasta")
        with open(individual_fasta, "w") as f:
            f.write(f">{sequence_id}\n{record.seq}\n")

        print(f"Processing {sequence_id}, Length: {sequence_length}")

        # Run full pipeline for this sequence
        try:
            psi_blast_output = run_psiblast(individual_fasta, sequence_output_dir)
            msa_fasta = extract_msa_from_psiblast(psi_blast_output, sequence_output_dir)
            aligned_msa = align_msa(msa_fasta, sequence_output_dir)
            gremlin_output = run_gremlin(aligned_msa, sequence_output_dir)
            dca_results = parse_gremlin_output(gremlin_output)
            contact_map = create_contact_map(dca_results, sequence_length)
            save_contact_map(contact_map, sequence_output_dir)
        except Exception as e:
            print(f"Error processing {sequence_id}: {e}")

fasta_file = "data/train/fasta/data.fasta"
make_blast_db(fasta_file=fasta_file, db_name=BLAST_DB)
process_fasta_file(fasta_file)
