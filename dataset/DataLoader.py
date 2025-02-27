import os
import torch
from torch.utils.data import Dataset
import numpy as np
from Bio import SeqIO


def custom_collate_fn(batch):
    batch = sorted(batch, key=lambda x: x['seq_length'], reverse=True)
    max_length = batch[0]['seq_length']

    esm2_embeddings_padded = []
    msa_embeddings_padded = []
    contact_maps_padded = []
    sequence_lengths = []
    masks_padded = []
    protein_ids = []

    for item in batch:
        L = item['seq_length']
        M = item['mask']
        P_id = item['id']
        sequence_lengths.append(L)
        protein_ids.append(P_id)

        # Pad esm2_embeddings
        esm2_emb = item['esm2_embeddings']
        pad_size = (0, 0, 0, max_length - L)
        esm2_emb_padded = torch.nn.functional.pad(esm2_emb, pad_size)  # Shape: (max_length, E_esm2)
        esm2_embeddings_padded.append(esm2_emb_padded)

        # Pad msa_embeddings (already mean-pooled across sequences)
        msa_emb = item['msa_embeddings']
        msa_emb_padded = torch.nn.functional.pad(msa_emb, pad_size)  # Shape: (max_length, E_msa)
        msa_embeddings_padded.append(msa_emb_padded)

        # Pad contact_map
        contact_map = item['contact_map']
        pad_size = (0, max_length - L, 0, max_length - L)
        contact_map_padded = torch.nn.functional.pad(contact_map, pad_size)
        contact_maps_padded.append(contact_map_padded)

        # Pad masks
        pad_size = (0, max_length - L)
        M_padded = torch.nn.functional.pad(M, pad_size)
        masks_padded.append(M_padded)

    # Stack tensors
    esm2_embeddings_batch = torch.stack(esm2_embeddings_padded)  # Shape: (batch_size, max_length, E_esm2)
    msa_embeddings_batch = torch.stack(msa_embeddings_padded)  # Shape: (batch_size, max_length, E_msa)
    contact_maps_batch = torch.stack(contact_maps_padded)  # Shape: (batch_size, max_length, max_length)
    sequence_lengths = torch.tensor(sequence_lengths)  # Shape: (batch_size)
    masks_padded = torch.stack(masks_padded) # Shape: (batch_size, max_length)


    return {
            'esm2_embeddings': esm2_embeddings_batch,
            'msa_embeddings': msa_embeddings_batch,
            'contact_map': contact_maps_batch,
            'mask': masks_padded,
            'seq_length': sequence_lengths,
            'id': protein_ids
        }


class ProteinContactMapDataset(Dataset):
    def __init__(self, fasta_file, embedding_dir, distance_map_dir, pdb_dir, msa_embedding_dir, msa_strings_dir):
        # Parse sequences and IDs from the single fasta file
        self.sequences = []
        self.protein_ids = []
        for record in SeqIO.parse(fasta_file, 'fasta'):
            self.protein_ids.append(record.id)
            self.sequences.append((record.id, str(record.seq)))

        # Load the multiple sequence alignment (MSA)
        # self.sequences = sorted(self.sequences, key=lambda x: len(x[1]))

        self.embedding_dir = embedding_dir
        self.distance_map_dir = distance_map_dir
        self.pdb_dir = pdb_dir
        self.msa_embedding_dir = msa_embedding_dir
        self.msa_strings_dir = msa_strings_dir

        # THESE Filtering are Optional. And they are slow.
        # If you are 100% sure that you don't have corrupted files, then just ignore this step
        # self._filter_sequences_with_msa_embeddings()
        # self.sequences = self.sequences[:100]
        # self.protein_ids = self.protein_ids[:100]
        # self._fileter_sequences_without_aligned_version()
        # self._filter_corrupted_contact_maps()

    def _filter_sequences_with_msa_embeddings(self):
        # Get the list of protein IDs that have corresponding MSA embeddings
        existing_msa_embeddings = {f.split('_')[0] for f in os.listdir(self.msa_embedding_dir) if
                                   f.endswith('_msa_embeddings.npz')}

        # Filter sequences to include only those with existing MSA embeddings
        self.sequences = [(id_, seq) for id_, seq in self.sequences if id_ in existing_msa_embeddings]
        self.protein_ids = [id_ for id_, seq in self.sequences]

    def _fileter_sequences_without_aligned_version(self):
        new_seq = []
        for protein_id, seq in self.sequences:
            msa_strings_file = os.path.join(self.msa_strings_dir, protein_id, "aligned_msa.fasta")
            msa_string = self.get_aligned_string(protein_id, msa_strings_file)
            if msa_string is None:
                continue
            new_seq.append((protein_id, seq))
        self.sequences = new_seq
        self.protein_ids = [id_ for id_, seq in self.sequences]

    def load_contact_map(self, protein_id):
        contact_map_file = os.path.join(self.distance_map_dir, f"contact_map_{protein_id}.npz")
        try:
            contact_map = np.load(contact_map_file)['contact_map']  # Shape: [seq_length, seq_length]
            return True
        except Exception as e:
            print(f"Error loading contact map for {protein_id}: {e}")
            return False

    def _filter_corrupted_contact_maps(self):
        new_seq = []
        for protein_id, seq in self.sequences:
            has_contact = self.load_contact_map(protein_id)
            if has_contact:
                new_seq.append((protein_id, seq))
            else:
                continue
        self.sequences = new_seq
        self.protein_ids = [id_ for id_, seq in self.sequences]

    def __len__(self):
        return len(self.sequences)

    def generate_mask(self, msa_string):
        return torch.tensor([1 if residue != '-' else 0 for residue in msa_string], dtype=torch.float32)

    def get_aligned_string(self, protein_id, msa_strings_file):
        for record in SeqIO.parse(msa_strings_file, 'fasta'):
            if str(record.id) == str(protein_id):
                return str(record.seq)
        return None

    def pad_esm2_embedding_with_mask(self, esm_embedding, msa_string, mask):
        L_msa, D_esm = len(msa_string), esm_embedding.shape[1]
        padded_embedding = torch.zeros((L_msa, D_esm), device=esm_embedding.device)
        non_gap_indices = torch.nonzero(mask).flatten()
        padded_embedding[non_gap_indices] = esm_embedding

        return padded_embedding

    def pad_contact_map_with_mask(self, contact_map, msa_string, mask):
        L_msa = len(msa_string) # Length of the MSA string (includes gaps)

        padded_contact_map = torch.zeros((L_msa, L_msa), device=contact_map.device)
        non_gap_indices = torch.nonzero(mask).squeeze()
        non_gap_indices = non_gap_indices.long()

        # Use advanced indexing to place the contact map into the padded contact map
        padded_contact_map[non_gap_indices.unsqueeze(1), non_gap_indices.unsqueeze(0)] = contact_map
        return padded_contact_map

    def __getitem__(self, idx):
        protein_id = self.sequences[idx][0]
        # sequence = self.sequences[idx][1]
        # sequence_length = len(sequence)

        # Load esm embedding
        embedding_file = os.path.join(self.embedding_dir, f"embedding_{protein_id}.npz")
        esm2_embeddings = np.load(embedding_file)['embedding']  # Shape: [seq_length, embedding_dim]
        esm2_embeddings = torch.from_numpy(esm2_embeddings)  # Convert to PyTorch tensor

        # Load msa embedding
        msa_embedding_file = os.path.join(self.msa_embedding_dir, f"{protein_id}_msa_embeddings.npz")
        msa_embeddings = np.load(msa_embedding_file)['msa_embeddings']
        msa_embeddings = torch.from_numpy(msa_embeddings).squeeze(0)
        msa_embeddings = torch.mean(msa_embeddings, dim=0, keepdim=True).squeeze(0)  # Shape: (aligned_seq_length, emb_size)

        # Load contact map
        contact_map_file = os.path.join(self.distance_map_dir, f"contact_map_{protein_id}.npz")
        contact_map = np.load(contact_map_file)['contact_map']  # Shape: [seq_length, seq_length]
        # contact_map = (contact_map < 8.0)
        contact_map = torch.from_numpy(contact_map).float()

        msa_strings_file = os.path.join(self.msa_strings_dir, str(protein_id), "aligned_msa.fasta")
        msa_string = self.get_aligned_string(protein_id, msa_strings_file)

        mask = self.generate_mask(msa_string)
        esm2_embeddings = self.pad_esm2_embedding_with_mask(esm2_embeddings, msa_string, mask)
        contact_map = self.pad_contact_map_with_mask(contact_map, msa_string, mask)

        return {
            'esm2_embeddings': esm2_embeddings,
            'msa_embeddings': msa_embeddings,
            'contact_map': contact_map,
            'mask': mask,
            'seq_length': len(mask),
            'id': protein_id
        }