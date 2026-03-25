from .modules import SAB, ISAB, PMA
from torch import nn


class SetTransformer(nn.Module):
    def __init__(
            self,
            dim_input,
            num_seeds,
            num_inds=32,
            dim_hidden=None,
            num_heads=4,
            dropout=0.0):
        super(SetTransformer, self).__init__()
        if dim_hidden is None:
            dim_hidden = dim_input
        self.enc = nn.Sequential(
            ISAB(dim_input, dim_hidden, num_heads, num_inds, dropout),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, dropout)
        ) # Encoder(X) = ISABm(ISABm(X))

        # Decoder consists of PMA followed by two SABs
        # In the original paper, the decoder also has a final rFF
        # after the SABs, but in our case this is handled by the
        # head MLP, so we omit it here for simplicity and efficiency.
        self.dec = nn.Sequential(
            PMA(dim_hidden, dim_hidden, num_heads, num_seeds, dropout=dropout),
            SAB(dim_hidden, dim_hidden, num_heads, dropout=dropout),
            SAB(dim_hidden, dim_hidden, num_heads, dropout=dropout),
        ) # SAB(PMAk(Z))

    def forward(self, x, mask=None):
        enc = x
        for layer in self.enc:
            enc = layer(enc, mask=mask)

        dec = enc
        for i, layer in enumerate(self.dec):
            dec = layer(dec, mask=mask if i == 0 else None) # Only apply mask in the PMA layer

        return enc, dec
    
def build_set_transformer(cfg, input_dim):
    # TODO: Add config options
    return SetTransformer(
        dim_input=input_dim,
        num_seeds=1,
        dropout=0.1
    )
    # return SetTransformer(
    #     dim_input=cfg.MODEL.SET_TRANSFORMER.DIM_INPUT,
    #     num_seeds=cfg.MODEL.SET_TRANSFORMER.NUM_SEEDS,
    #     num_inds=cfg.MODEL.SET_TRANSFORMER.NUM_INDS,
    #     dim_hidden=cfg.MODEL.SET_TRANSFORMER.DIM_HIDDEN,
    #     num_heads=cfg.MODEL.SET_TRANSFORMER.NUM_HEADS,
    #     dropout=cfg.MODEL.SET_TRANSFORMER.DROPOUT
    # )