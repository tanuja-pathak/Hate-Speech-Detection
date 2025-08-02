import torch.nn as nn

class TweetLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size + 1, 
            embedding_dim=embedding_dim, 
            padding_idx=padding_idx
        )
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim//2, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_dim//2, hidden_dim//2, batch_first=True)
        self.fc = nn.Linear(hidden_dim//2, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        x, _ = self.lstm1(embedded)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x = x[:, -1, :]  # Last timestep
        return self.fc(x)