import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context):
        h = self.heads

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: t.reshape(t.shape[0], -1, h, t.shape[-1] // h).transpose(1, 2), (q, k, v))

        # Compute attention scores in chunks to save memory
        chunk_size = 128  # Adjust this based on your GPU memory
        attn_chunks = []
        for i in range(0, q.shape[2], chunk_size):
            attn_chunk = torch.einsum('bhid,bhjd->bhij', q[:, :, i:i+chunk_size], k) * self.scale
            attn_chunk = F.softmax(attn_chunk, dim=-1)
            attn_chunks.append(attn_chunk)

        attn = torch.cat(attn_chunks, dim=2)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.transpose(1, 2).reshape(out.shape[0], -1, out.shape[-1] * h)
        return self.to_out(out)