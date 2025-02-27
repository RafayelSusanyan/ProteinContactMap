import torch
import torch.nn as nn

# Just function that might be helpful, but didn't have time to use them.
# def compute_precision_at_Lk(predicted_map, true_map, k):
#     L = predicted_map.shape[0]
#     N = L // k
#
#     # Exclude diagonal and lower triangle
#     triu_indices = torch.triu_indices(L, L, offset=1)
#     predicted_scores = predicted_map[triu_indices[0], triu_indices[1]]
#     true_contacts = true_map[triu_indices[0], triu_indices[1]]
#
#     # Get top N predictions
#     top_n_indices = torch.topk(predicted_scores, N).indices
#     top_n_true_contacts = true_contacts[top_n_indices]
#
#     # Compute precision
#     precision = top_n_true_contacts.sum().item() / N
#     return precision


class ContactMapPredictor(nn.Module):
    def __init__(self, esm_dim=1280, msa_dim=768, dropout_rate=0.2):
        super(ContactMapPredictor, self).__init__()
        self.esm_dim = esm_dim
        self.msa_dim = msa_dim
        self.hidden_dim = esm_dim + msa_dim

        # Layer to learn the weight matrix W
        self.W = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        nn.init.xavier_uniform_(self.W)
        print("IS W TRAINABLE? ", self.W.requires_grad)

        # Convolutional layers or feedforward network
        # Example using a simple CNN
        # Note we didn't use here the BatchNormalization, because we trained only with batch size = 1 (GPU memory)
        # We can try also to add more layers
        # We can try also to add dropout
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # Input channels = 1
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1)  # Output channels = 1
        )
        # self.cnn_layers = nn.Sequential(
        #     nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Input channels = 1
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 1, kernel_size=1) # Output channels = 1
        # )

    def forward(self, esm_embedding, msa_embeddings):
        """
        Args:
            esm_embedding (torch.Tensor): Tensor of shape (seq_len, 1280)
            msa_embeddings (torch.Tensor): Tensor of shape (n_seq, seq_len, 768)

        Returns:
            contact_map (torch.Tensor): Tensor of shape (seq_len, seq_len)
        """
        concatenated_embedding = torch.cat((esm_embedding, msa_embeddings), dim=2)  # Shape: (batch_size, seq_len, 2048)

        # Step 1: Compute the score matrix
        # scores = A @ W @ A^T
        scores = torch.matmul(concatenated_embedding, self.W)  # Shape: (batch_size, seq_len, 2048)
        scores = torch.matmul(scores, concatenated_embedding.transpose(1, 2))  # Shape: (batch_size, seq_len, seq_len)

        # Step 2: Apply ReLU activation
        scores = torch.relu(scores)  # Shape: (batch_size, seq_len, seq_len)
        scores = scores.unsqueeze(1)  # Shape: (batch_size=1, channels=1, seq_len, seq_len)

        # Step 3: Apply CNN
        contact_map = self.cnn_layers(scores)  # Shape: (batch_size, 1, seq_len, seq_len)
        contact_map = contact_map.squeeze(1)  # Shape: (batch_size, seq_len, seq_len)

        # Step 4: sum with scores and after sigmoid activation
        # might be better to have several residual blocks as in the paper
        # One might try small network withot residual technique, it actually gave better results on small set then the residual version
        scores = scores.squeeze(1)
        contact_map += scores

        # Maybe here before sigmoid we can add another residual block
        contact_map = torch.sigmoid(contact_map)
        return contact_map
