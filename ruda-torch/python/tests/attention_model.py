import torch


class AttentionBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.LayerNorm(8)
        self.query = torch.nn.Linear(8, 8)
        self.key = torch.nn.Linear(8, 8)
        self.value = torch.nn.Linear(8, 8)
        self.projection = torch.nn.Linear(8, 8)
        self.ffn = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.SiLU(), torch.nn.Linear(16, 8))

    def forward(self, x):
        batch, sequence, _ = x.shape
        normalized = self.norm(x)
        q = self.query(normalized).view(batch, sequence, 2, 4).transpose(1, 2)
        k = self.key(normalized).view(batch, sequence, 2, 4).transpose(1, 2)
        v = self.value(normalized).view(batch, sequence, 2, 4).transpose(1, 2)
        probabilities = torch.softmax((q @ k.transpose(-2, -1)) / 2, dim=-1)
        context = (probabilities @ v).transpose(1, 2).reshape(batch, sequence, 8)
        residual = x + self.projection(context)
        return residual + self.ffn(residual)


class BuiltinAttentionBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.LayerNorm(8)
        self.attention = torch.nn.MultiheadAttention(8, 2, dropout=0, batch_first=True)
        self.ffn = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.SiLU(), torch.nn.Linear(16, 8))

    def forward(self, x):
        normalized = self.norm(x)
        attention, _ = self.attention(normalized, normalized, normalized)
        residual = x + attention
        return residual + self.ffn(residual)
