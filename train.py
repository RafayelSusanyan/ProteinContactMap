import torch
from torch.utils.data import DataLoader, random_split, Subset
from dataset.DataLoader import ProteinContactMapDataset
from dataset.DataLoader import custom_collate_fn
from models.ContactPredictor import ContactMapPredictor
import os
from dotenv import load_dotenv
from tqdm import tqdm
import pickle

# Load environment variables
load_dotenv()

# Define paths from environment variables
TRAIN_FASTA_PATH = os.getenv('TRAIN_FASTA_PATH')
TRAIN_EMBEDDING_DIR = os.getenv('TRAIN_EMBEDDING_DIR')
TRAIN_DISTANCE_MAP_DIR = os.getenv('TRAIN_DISTANCE_MAP_DIR')
TRAIN_PDB_DIR = os.getenv('TRAIN_PDB_DIR')
TRAIN_MSA_EMBEDDING_DIR = os.getenv('TRAIN_MSA_EMBEDDING_DIR')
TRAIN_MSA_STRINGS_DIR = os.getenv('TRAIN_MSA_STRINGS_DIR')

# Directory to save model checkpoints
CHECKPOINT_DIR = os.getenv('CHECKPOINT_DIR', './checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# File path to save the best model
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'best_model.pth')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
esm2_dim = int(os.getenv('ESM_EMBEDDING_DIMENSION'))
msa_dim = int(os.getenv('MSA_EMBEDDING_DIMENSION'))
num_epochs = int(os.getenv('NUM_EPOCHS'))
batch_size = int(os.getenv('BATCH_SIZE'))
validation_split = 0.1  # Percentage of data to use for validation

# Initialize model
model = ContactMapPredictor(esm2_dim, msa_dim)
model = model.to(device)

# Initialize dataset
dataset = ProteinContactMapDataset(
    fasta_file=TRAIN_FASTA_PATH,
    embedding_dir=TRAIN_EMBEDDING_DIR,
    distance_map_dir=TRAIN_DISTANCE_MAP_DIR,
    pdb_dir=TRAIN_PDB_DIR,
    msa_embedding_dir=TRAIN_MSA_EMBEDDING_DIR,
    msa_strings_dir=TRAIN_MSA_STRINGS_DIR
)

# Split dataset into training and validation sets
dataset_size = len(dataset)
val_size = int(validation_split * dataset_size)
train_size = dataset_size - val_size

# if you want to split random, do this. Here I don't split random because I'm running on same train-val several times.
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_dataset = Subset(dataset, range(train_size))
val_dataset = Subset(dataset, range(train_size, train_size + val_size))

# Initialize dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

# Define loss function and optimizer
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=5e-4)

# Initialize variables for checkpoint loading
start_epoch = 0
train_losses = []
val_losses = []
best_val_loss = float('inf')

checkpoints = [int(i.split('_')[1].split('.')[0]) for i in os.listdir(CHECKPOINT_DIR) if i.startswith('checkpoint_')]

# Check if checkpoint exists
if len(checkpoints) > 0:
    last_ckpt = max(checkpoints)
    CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, f"checkpoint_{last_ckpt}.pth")
    print(f"Loading checkpoint from {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    best_val_loss = checkpoint['best_val_loss']
    print(f"Resuming training from epoch {start_epoch}")
else:
    print("No checkpoint found. Starting training from scratch.")

end_epoch = start_epoch + num_epochs
# Training loop
for epoch in range(start_epoch, end_epoch):
    # Training phase
    model.train()
    total_train_loss = 0

    # Wrap the dataloader with tqdm for progress bar
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{end_epoch} - Training")

    for batch in progress_bar:
        esm2_embeddings = batch['esm2_embeddings'].to(device)
        msa_embeddings = batch['msa_embeddings'].to(device)
        contact_maps = batch['contact_map'].to(device)

        optimizer.zero_grad()

        # Forward pass
        predictions = model(esm2_embeddings, msa_embeddings)

        # Compute loss
        loss = criterion(predictions, contact_maps)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

        # Update progress bar
        progress_bar.set_postfix(loss=loss.item())

    avg_train_loss = total_train_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)  # Append training loss
    print(f"Epoch [{epoch+1}/{end_epoch}], Average Training Loss: {avg_train_loss:.4f}")

    # Validation phase
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            esm2_embeddings = batch['esm2_embeddings'].to(device)
            msa_embeddings = batch['msa_embeddings'].to(device)
            contact_maps = batch['contact_map'].to(device)

            # Forward pass
            predictions = model(esm2_embeddings, msa_embeddings)

            # Compute loss
            loss = criterion(predictions, contact_maps)

            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_dataloader)
    val_losses.append(avg_val_loss)  # Append validation loss
    print(f"Epoch [{epoch+1}/{end_epoch}], Average Validation Loss: {avg_val_loss:.4f}")

    # Save the model checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
    }

    if epoch % 5 == 0:
        CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, f'checkpoint_{epoch}.pth')
        torch.save(checkpoint, CHECKPOINT_PATH)
        losses_dict = {'train_losses': train_losses, 'val_losses': val_losses}
        with open(os.path.join(CHECKPOINT_DIR, 'losses.pkl'), 'wb') as f:
            pickle.dump(losses_dict, f)
        print(f"Checkpoint saved to {CHECKPOINT_PATH}")

    # Save the best model separately
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"Best model updated and saved to {BEST_MODEL_PATH}")

# After training, you can plot the losses
epochs = range(1, len(train_losses) + 1)


# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 5))
# plt.plot(epochs, train_losses, 'b-', label='Training Loss')
# plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
# plt.title('Training and Validation Loss over Epochs')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Optionally, save the plot
# plt.savefig(os.path.join(CHECKPOINT_DIR, 'loss_curve.png'))

# Save the losses to a file for future use