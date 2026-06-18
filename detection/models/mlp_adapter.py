# external MLP adapter based on Ibsen et al.
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPAdapter(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.LeakyReLU(0.2),

            nn.Linear(embedding_size, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.LeakyReLU(0.2),

            nn.Linear(embedding_size, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.LeakyReLU(0.2),

            nn.Linear(embedding_size, embedding_size),
            nn.BatchNorm1d(embedding_size),
        )

    def forward(self, emb):
        out = self.model(emb)
        out = F.normalize(out, p=2, dim=1)
        return out