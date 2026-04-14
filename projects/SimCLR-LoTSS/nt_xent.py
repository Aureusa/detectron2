import torch
import torch.nn.functional as F


def nt_xent_loss(z1, z2, temperature=0.5):
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)           # (2B, D)
    sim = torch.mm(z, z.T) / temperature      # (2B, 2B)
    
    # Mask out self-similarity
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float('-inf'))
    
    # Positive pairs: (i, i+B) and (i+B, i)
    labels = torch.cat([
        torch.arange(B, 2 * B),
        torch.arange(0, B)
    ]).to(z.device)
    
    return F.cross_entropy(sim, labels)
