from torch import nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F


class DoubleAttentionLayer(nn.Module):
    """
    Implementation of Double Attention Network. NIPS 2018
    """
    def __init__(self, in_channels: int, c_m: int, c_n: int):
        """

        Parameters
        ----------
        in_channels
        c_m
        c_n
        """
        super(DoubleAttentionLayer, self).__init__()
        self.c_m = c_m
        self.c_n = c_n
        self.in_channels = in_channels
        self.convA = nn.Conv2d(in_channels, c_m, kernel_size = 1)
        self.convB = nn.Conv2d(in_channels, c_n, kernel_size = 1)
        self.convV = nn.Conv2d(in_channels, c_n, kernel_size = 1)

    def forward(self, x: torch.Tensor):
        """

        Parameters
        ----------
        x: `torch.Tensor` of shape (B, C, H, W)

        Returns
        -------

        """
        batch_size, c, h, w = x.size()
        assert c == self.in_channels, 'input channel not equal!'
        A = self.convA(x)  # (B, c_m, h, w) because kernel size is 1
        B = self.convB(x)  # (B, c_n, h, w)
        V = self.convV(x)  # (B, c_n, h, w)
        tmpA = A.view(batch_size, self.c_m, h * w)
        attention_maps = B.view(batch_size, self.c_n, h * w)
        attention_vectors = V.view(batch_size, self.c_n, h * w)
        attention_maps = F.softmax(attention_maps, dim = -1)  # softmax on the last dimension to create attention maps
        # step 1: feature gathering
        global_descriptors = torch.bmm(tmpA, attention_maps.permute(0, 2, 1))  # (B, c_m, c_n)
        # step 2: feature distribution
        attention_vectors = F.softmax(attention_vectors, dim = -1)  # (B, c_n, h * w)
        tmpZ = global_descriptors.matmul(attention_vectors)  # B, self.c_m, h * w
        return tmpZ
