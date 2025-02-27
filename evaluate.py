import torch
from torch.utils.data import DataLoader
from dataset.DataLoader import ProteinContactMapDataset
from dataset.DataLoader import custom_collate_fn
from models.ContactPredictor import ContactMapPredictor
import os
from dotenv import load_dotenv
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef
import matplotlib.pyplot as plt
import numpy as np

# Load environment variables
_ = load_dotenv()

# Define paths from environment variables
TEST_FASTA_PATH = os.getenv('TEST_FASTA_PATH')
TEST_EMBEDDING_DIR = os.getenv('TEST_EMBEDDING_DIR')
TEST_DISTANCE_MAP_DIR = os.getenv('TEST_DISTANCE_MAP_DIR')
TEST_PDB_DIR = os.getenv('TEST_PDB_DIR')
TEST_MSA_EMBEDDING_DIR = os.getenv('TEST_MSA_EMBEDDING_DIR')
TEST_MSA_STRINGS_DIR = os.getenv('TEST_MSA_STRINGS_DIR')

# Directory and file paths
CHECKPOINT_DIR = os.getenv('CHECKPOINT_DIR', './checkpoints')
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'best_model.pth')

# Directory to save contact map plots
PLOTS_DIR = os.path.join(CHECKPOINT_DIR, 'contact_map_plots')
os.makedirs(PLOTS_DIR, exist_ok=True)  # Create the directory if it doesn't exist

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
esm2_dim = int(os.getenv('ESM_EMBEDDING_DIMENSION'))
msa_dim = int(os.getenv('MSA_EMBEDDING_DIMENSION'))
batch_size = 1  # Process one protein at a time due to varying sizes

# Initialize model
model = ContactMapPredictor(esm2_dim, msa_dim)
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()  # Set the model to evaluation mode

# Initialize test dataset
test_dataset = ProteinContactMapDataset(
    fasta_file=TEST_FASTA_PATH,
    embedding_dir=TEST_EMBEDDING_DIR,
    distance_map_dir=TEST_DISTANCE_MAP_DIR,
    pdb_dir=TEST_PDB_DIR,
    msa_embedding_dir=TEST_MSA_EMBEDDING_DIR,
    msa_strings_dir=TEST_MSA_STRINGS_DIR
)

# Initialize dataloader
test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=custom_collate_fn
)

# Define loss function
criterion = torch.nn.BCELoss()

# Initialize variables for evaluation metrics
total_loss = 0.0
all_predictions = []
all_targets = []
num_evaluated = 0  # Counter for proteins evaluated
threshold = 0.15

# Evaluation loop
with torch.no_grad():
    progress_bar = tqdm(test_dataloader, desc="Evaluating")
    for idx, batch in enumerate(progress_bar):
        esm2_embeddings = batch['esm2_embeddings'].to(device)
        msa_embeddings = batch['msa_embeddings'].to(device)
        contact_maps = batch['contact_map'].to(device)
        masks = batch['mask'].to(device)  # Shape: (batch_size, max_length)

        # Forward pass
        predictions = model(esm2_embeddings, msa_embeddings)

        # Extract sequence length
        seq_lengths = batch['seq_length']

        for i in range(batch_size):
            L = seq_lengths[i].item()
            pred_map = predictions[i][:L, :L]
            true_map = contact_maps[i][:L, :L]
            mask = masks[i][:L]
            protein_id = batch['id'][i]  # Corrected key to 'protein_id'

            # Get indices where mask is 1
            indices = torch.nonzero(mask, as_tuple=False).squeeze(-1).long()

            if indices.numel() == 0:
                print(f"No valid residues for protein {protein_id}, skipping.")
                continue  # Skip if no valid residues

            # Extract corresponding submatrices
            submatrix_pred = pred_map[indices.unsqueeze(0), indices.unsqueeze(1)]
            submatrix_true = true_map[indices.unsqueeze(0), indices.unsqueeze(1)]

            # Apply threshold to convert predictions to binary values
            submatrix_pred_binary = (submatrix_pred >= threshold).float()

            # Flatten and collect predictions and targets
            all_predictions.append(submatrix_pred_binary.cpu().flatten())
            all_targets.append(submatrix_true.cpu().flatten())

            # Compute loss
            loss = criterion(submatrix_pred_binary, submatrix_true)
            total_loss += loss.item()
            num_evaluated += 1  # Increment counter

            # Convert to NumPy for plotting
            pred_map_np = submatrix_pred_binary.cpu().numpy()
            true_map_np = submatrix_true.cpu().numpy()
            print(f"Predicted Contact Map - Non-zero entries: {np.count_nonzero(pred_map_np)}")
            print(f"True Contact Map - Non-zero entries: {np.count_nonzero(true_map_np)}")
            print(true_map_np)
            print(pred_map_np)
            # Save the contact map plots
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            # Plot True Contact Map
            axes[0].imshow(true_map_np, cmap='Blues', interpolation='nearest')
            axes[0].set_title(f"True Contact Map ({protein_id})")
            axes[0].set_xlabel("Residue Index")
            axes[0].set_ylabel("Residue Index")

            # Plot Predicted Contact Map
            axes[1].imshow(pred_map_np, cmap='Blues', interpolation='nearest')
            axes[1].set_title(f"Predicted Contact Map ({protein_id})")
            axes[1].set_xlabel("Residue Index")
            axes[1].set_ylabel("Residue Index")

            plt.tight_layout()
            plot_filename = os.path.join(PLOTS_DIR, f"contact_map_{protein_id}.png")
            plt.savefig(plot_filename)
            plt.close(fig)  # Close the figure to free memory

    # Compute average loss
    if num_evaluated > 0:
        avg_loss = total_loss / num_evaluated
    else:
        avg_loss = float('nan')
    print(f"\nAverage Loss on Test Set: {avg_loss:.4f}")

    # Concatenate all predictions and targets
    if all_predictions:
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)

        # Apply threshold to get binary predictions
        binary_predictions = (all_predictions >= threshold).float()

        # Convert tensors to numpy arrays
        y_true = all_targets.numpy()
        y_pred = binary_predictions.numpy()
        y_scores = all_predictions.numpy()

        # Compute evaluation metrics
        accuracy = (y_pred == y_true).mean()
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)

        print(f"Accuracy on Test Set: {accuracy * 100:.2f}%")
        print(f"Precision on Test Set: {precision * 100:.2f}%")
        print(f"Recall on Test Set: {recall * 100:.2f}%")
        print(f"F1 Score on Test Set: {f1 * 100:.2f}%")
        print(f"MCC on Test Set: {mcc:.4f}")
    else:
        print("No valid data to compute evaluation metrics.")
