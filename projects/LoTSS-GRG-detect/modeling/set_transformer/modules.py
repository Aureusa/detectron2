import torch
import torch.nn as nn
import torch.nn.functional as F
import math

    
class rFF(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.0):
        super(rFF, self).__init__()
        hidden_dim = dim * 4 if hidden_dim is None else hidden_dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x
    
    
class MAB(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=None, num_heads=4, dropout=0.0):
        super(MAB, self).__init__()
        self.multi_head = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.rff = rFF(embedding_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, y, mask=None):
        attention, _ = self.multi_head(query=self.ln1(x), key=self.ln1(y), value=self.ln1(y), key_padding_mask=mask) 
        H = x + self.dropout1(attention)
        H = H + self.dropout2(self.rff(self.ln2(H)))
        return H
    

class SAB(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=None, num_heads=4, dropout=0.0):
        super(SAB, self).__init__()
        self.mab = MAB(embedding_dim, hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)

    def forward(self, x, mask=None):
        return self.mab(x, x, mask=mask)


class ISAB(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=None, num_heads=4, num_inds=32, dropout=0.0):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, embedding_dim))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(embedding_dim, hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        self.mab1 = MAB(embedding_dim, hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)

    def forward(self, x, mask=None):
        H = self.mab0(self.I.repeat(x.size(0), 1, 1), x, mask=mask)
        # No mask in the second attention
        # TODO: Maybe fix some other time if this turns out to be a problem,
        # but for now we want the second attention to be able to attend
        # to all the induced points regardless of the input mask.
        return self.mab1(x, H, mask=None)
    

class PMA(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=None, num_heads=4, num_seeds=1, dropout=0.0):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, embedding_dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(embedding_dim, hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
        self.rff = rFF(embedding_dim, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, x, mask=None):
        rff = self.rff(x)
        S = self.S.repeat(x.size(0), 1, 1)
        return self.mab(S, rff, mask=mask)
    