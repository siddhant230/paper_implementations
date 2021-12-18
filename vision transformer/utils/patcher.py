import torch.nn as nn


class PatchEmbedder(nn.Module):

    """
     Splits image into smaller patches and embed them

     PARAMETRS
     -----------
     image_size : int
        size of the image

     patch_size : int
        size of patches to embed

     input_channels : int
        number of input channels

     embed_dims : int
        The embedding dimension

     ATTRIBUTES
     ----------
     n_patches : int
        total number of patches formed

     projection : nn.Conv2D
        Conv layer that breaks image into patches and then embeds

    """

    def __init__(self, image_size, patch_size,
                 input_channels=3, embed_dims=512):

        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.input_channels = input_channels
        self.embed_dims = embed_dims
        self.n_patches = (self.image_size // self.patch_size) ** 2

        self.projection = nn.Conv2d(
            input_channels,
            self.embed_dims,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

    def forward(self, x):
        """
        Forward pass : breaks into patches and them flattens

        PARAMETERS
        ----------
        x : torch.Tensor
            input tensor image
            shape : (n_samples, input_channels, image_size, image_size)

        RETURNS
        -------
        out : torch.Tensor
            output tensor : patched + embedded
            shape : (n_samples, n_patches, embed_dims)
        """
        x = self.projection(
            x)  # (n_samples, embed_dims, n_patches**0.5, n_patches**0.5)
        # x -> (100, 512, 16, 16)
        # we need to flatten this
        out = x.flatten(2)  # (n_samples, embed_dims, n_patches)
        # out -> (100, 512, 256)
        out = out.transpose(1, 2)
        # out -> (100, 256, 512)
        return out
