import torch.nn as nn


class MLP(nn.Module):
    """
    MLP module

    PARAMETERS
    ----------
    in_features : int
        Number of input features

    hidden_features : int
        Number of hidden features

    out_features : int
        Number of output features

    p : float
        Dropout probability

    ATTRIBUTES
    ----------
    fc1 : nn.Linear
        Linear layer 1

    activation : nn.Gelu
        Gelu activation

    fc2 : nn.Linear
        Linear layer 2

    drop : nn.Dropout
        Dropout layer
    """

    def __init__(self, in_features, out_features, hidden_features, p=0.5):

        super().__init__()

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        """
        Forward pass

        PARAMETERS
        ----------
        x : torch.Tensor
            input tensor : (n_samples, n_patches+1, in_features)
            (100, 16+1, 512)    1 extra patch is for CLS token

        RETURNS
        -------
        out : torch.Tensor
            output tensor : (n_samples, n_patches+1, out_features)
            (100, 16+1, 512)    1 extra patch is for CLS token
        """

        x = self.fc1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.fc2(x)
        out = self.drop(x)
        return out
