import torch
import torch.nn as nn

class SimpleTextEncoder(nn.Module):
    def __init__(self, vocab_size=1000, max_length=77, embedding_dim=768, num_layers=6):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_length, embedding_dim)
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8, dim_feedforward=2048, dropout=0.1)
            for _ in range(num_layers)
        ])
        
        self.ln_final = nn.LayerNorm(embedding_dim)
        self.vocab = {chr(i+97): i for i in range(26)}  # a-z
        self.vocab[' '] = 26  # space
        self.max_length = max_length

    def tokenize(self, text):
        return [self.vocab.get(c, len(self.vocab)) for c in text.lower()][:self.max_length]

    def forward(self, text_list):
        if text_list is None:
            # Return a tensor of zeros if there's no text input
            return torch.zeros(1, self.max_length, 768, device=self.token_embedding.weight.device)
        
        tokens = [self.tokenize(text) for text in text_list]
        tokens = [t + [len(self.vocab)]*(self.max_length - len(t)) for t in tokens]  # Pad with [len(self.vocab)]
        tokens = torch.tensor(tokens).to(self.token_embedding.weight.device)
        
        batch_size, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
        
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.ln_final(x)
        return x