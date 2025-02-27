import os.path
import subprocess

def make_blast_db(fasta_file, db_name):
    db_files = [f"{db_name}.pin", f"{db_name}.psq", f"{db_name}.phr"]
    if all(os.path.isfile(db_file) for db_file in db_files):
        print(f"BLAST database '{db_name}' already exists. Skipping creation.")
        return
    subprocess.run(['makeblastdb', '-in', fasta_file, '-dbtype', 'prot', '-out', db_name, '-parse_seqids'])

def run_blast(query_sequence, db_name, out_file, protein_id, blast_dir, evalue=1e-5, max_target_seqs=100):
    # Write query sequence to a temporary file
    if os.path.exists(out_file):
        return
    else:
        # Write query sequence to a temporary file
        file_path = os.path.join(blast_dir, f'{protein_id}.fasta')
        with open(file_path, 'w') as f:
            f.write(f'>query\n{query_sequence}')
        # Run BLAST
        subprocess.run([
            'blastp',
            '-query', file_path,
            '-db', db_name,
            '-out', out_file,
            '-outfmt', '6',
            '-evalue', str(evalue),
            '-max_target_seqs', str(max_target_seqs)
        ])
        # Remove temporary file
        os.remove(file_path)

def parse_blast_results(out_file, identity_threshold=80.0):
    similar_sequences = []
    with open(out_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            subject_id = parts[1]
            identity = float(parts[2])
            if identity >= identity_threshold:
                similar_sequences.append(subject_id)
    return similar_sequences