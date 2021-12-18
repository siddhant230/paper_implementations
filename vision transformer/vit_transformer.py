import torch
import torch.nn as nn
from utils.vit_blocks import Block
from utils.patcher import PatchEmbedder


class VisionTransformer(nn.Module):
    """
    Simple ViT

    PARAMETERS
    ----------
    image_size : int
        size of input image

    patch_size : int
        size of each patch

    input_channels : int
        number of input channels

    n_classes : int
        number of classes

    embed_dim : int
        embedding dimensions

    depth : int
        total number of blocks

    n_heads : int
        number of attention heads

    mlp_ratio : float
        dims of MLP wrt to dim

    qkv_bias : bool
        to include bias on linear projection

    p, att_p : float
        dropout probability

    ATTRIBUTES
    ----------
    patch_embed : PatchEmbedder
        patch embedding layer

    cls_token : nn.Parameter
        First token in sequence  : learnable

    pos_embed : nn.Parameter
        positional embedding of CLS + all patches (1 + 16)

    pos_drop : nn.Dropout
        Dropout layer

    blocks : nn.ModuleList
        list of all Block Modules

    norm : nn.LayerNorm
        Layer normalization
    """

    # torch.Size([1, 577, 512]) torch.Size([1, 257, 512])

    def __init__(self, image_size=256,
                 patch_size=16,
                 input_channels=3,
                 n_classes=10,
                 embed_dim=512,
                 depth=12,
                 n_heads=16,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 p=0.0,
                 att_p=0.0):

        super().__init__()
        self.patch_embedding = PatchEmbedder(image_size=image_size,
                                             patch_size=patch_size,
                                             input_channels=input_channels,
                                             embed_dims=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(
            1, 1 + self.patch_embedding.n_patches, embed_dim))
        self.blocks = nn.ModuleList(
            [
                Block(dim=embed_dim,
                      n_heads=n_heads,
                      mlp_ratio=mlp_ratio,
                      qkv_bias=qkv_bias,
                      p=p, att_p=att_p)

                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        """
        Forward pass

        PARAMETERS
        ----------
        x : torch.Tensor
            input tensor : (n_samples, input_channels, image_size, image_size)

        RETURNS
        -------
        out_logits : torch.Tensor
            Logits over all the classes,
            output tensor : (n_samples, n_classes)
        """
        n_samples = x.shape[0]
        x = self.patch_embedding(x)
        print(x.shape)

        cls_token = self.cls_token.expand(n_samples, -1, -1)
        # (n_samples, embed_dim) -> (n_samples, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1)
        # (n_samples,n_patches,embed_dim) -> (n_samples,1+n_patches,embed_dim)
        x = x + self.pos_embed

        # input prepared
        for each_block in self.blocks:
            x = each_block(x)

        x = self.norm(x)
        cls_token_final = x[:, 0]  # all cls tokens
        out_logits = self.head(cls_token_final)
        return out_logits


if __name__ == "__main__":
    image_size = 256  # 384
    embed_dim = 512  # 768
    n_heads = 16   # 12
    vit_obj_model = VisionTransformer(
        image_size=image_size, embed_dim=embed_dim, n_heads=n_heads)

    print(vit_obj_model)
    torch.save(vit_obj_model, "vit_test_model.pth")

    vit_obj_model.eval()
    x = torch.zeros((1, 3, image_size, image_size))
    logits = vit_obj_model(x)
    print(logits.shape)
    print(logits)
