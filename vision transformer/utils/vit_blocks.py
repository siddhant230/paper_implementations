import torch.nn as nn
from utils.attention import Attention
from utils.mlp import MLP


class Block(nn.Module):

    """
    Transformer block

    PARAMETERS
    ----------
    dim : int
        Embedding dimension

    n_heads : int
        Number of attention heads

    mlp_ratio : float
        dims of MLP wrt to dim

    qkv_bias : bool
        bias to include for qkv

    p, att_p : float
        dropout probability

    ATTRIBUTES
    ----------
    norm1, norm2 : nn.LayerNomrs
        Layer Normalization

    attention : Attention
        attention blocks

    mlp : MLP
        mlp block
    """

    def __init__(self, dim, n_heads, mlp_ratio=4.0,
                 qkv_bias=False, p=0.0, att_p=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn_block = Attention(
            dims=dim,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            attention_drop_p=att_p,
            projection_drop_p=p,
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_ftrs = int(dim * mlp_ratio)
        self.mlp_block = MLP(
            in_features=dim,
            out_features=dim,
            hidden_features=hidden_ftrs
        )

    def forward(self, x):
        """
        Forward pass

        PARAMETERS
        ----------
        x : torch.Tensor
            input tensor : (n_samples, n_patches+1, dim)

        RETURNS
        -------
        out : torch.Tensor
            output tensor : (n_samples, n_patches+1, dim)
        """

        # residual block 1 : norm + attention
        x = x + self.attn_block(self.norm1(x))
        # residual block 2 : norm + mlp
        out = x + self.mlp_block(self.norm2(x))

        return out
