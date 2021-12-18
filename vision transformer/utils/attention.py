import torch.nn as nn


class Attention(nn.Module):

    """
    Attention block/mechanism

    PARAMETRS
    ---------
    dims : int
        input and output dimensions

    n_heads : int
        number of attention blocks

    attention_drop_p : int
        Dropout probab on Q, K and V

    projection_drop_p : int
        Dropout probab for output tensor

    qkv_bias : bool
        to include bias on linear projection


    ATTRIBUTES
    ----------
    scale : float
        Normalization factor for dot product

    qkv : nn.Linear
        Linear projection of query, key and value tensors

    projection : nn.Linear
        Linear map; takes output from all attention heads and
        projects onto a new space

    attention_dropout, projection_dropout : nn.Dropout
        Normal dropout layers

    """

    def __init__(self, dims, n_heads, qkv_bias=False,
                 attention_drop_p=0.5, projection_drop_p=0.5):
        super().__init__()

        self.dims = dims
        self.n_heads = n_heads
        self.qkv_bias = qkv_bias
        self.attention_drop_p = attention_drop_p
        self.projection_drop_p = projection_drop_p

        self.head_dim = dims // n_heads
        self.scale = self.head_dim ** -0.5  # scaling factor before softmax,
        # to not allow very large values to flow into softmax,
        # that might result into smaller gradients.

        # this will generate query, key, value triplet
        self.qkv = nn.Linear(self.dims, self.dims*3, bias=self.qkv_bias)
        self.attention_dropout = nn.Dropout(attention_drop_p)
        self.projection = nn.Linear(self.dims, self.dims)
        self.projection_dropout = nn.Dropout(projection_drop_p)

    def forward(self, x):
        """
        Forward pass

        PARAMETERS
        ----------
        x : torch.Tensor
            input tensor : (n_samples, n_patches+1, embed_dims)
            (100, 256+1, 512)    1 extra patch is for CLS token

        RETURNS
        -------
        out : torch.Tensor
            output tensor : (n_samples, n_patches+1, embed_dims)
            (100, 256+1, 512)    1 extra patch is for CLS token
        """
        n_samples, n_tokens, dims = x.shape

        if dims != self.dims:
            raise ValueError("dimensions mismatch")
        qkv = self.qkv(x)
        # for qkv
        # input -> (100, 256+1, 512), output -> (100, 256+1, 512 * 3)
        qkv = qkv.reshape(n_samples, n_tokens, 3,
                          self.n_heads, self.head_dim
                          ).permute(2, 0, 3, 1, 4)
        # before reshape -> (100, 256+1, 512*3)
        # after reshape -> (100, 256+1, 3, n_heads, 512//)
        # after permutation -> (3, 100, n_heads, 256+1, 512)
        query, key, value = qkv[0], qkv[1], qkv[2]
        dot_prouct_q_k = (query @ key.transpose(-2, -1)) * self.scale
        attention = dot_prouct_q_k.softmax(dim=-1)
        attention = self.attention_dropout(attention)
        # attention output -> (100, n_heads, 256+1, 256+1)
        context = attention @ value
        context = context.transpose(1, 2).flatten(2)
        out = self.projection(context)
        out = self.projection_dropout(out)

        return out
