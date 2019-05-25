import torch
import torch.nn as nn

from nn_utils import conv1x1


class GlobalAttentionGeneral(nn.Module):
    """
    Global attention takes a matrix and a query matrix.
    Based on each query vector q, it computes a parameterized convex combination of the matrix
    H_1 H_2 H_3 ... H_n
      q   q   q       q
        |  |   |       |
          \ |   |      /
                  .....
              \   |  /
                      a
    Constructs a unit mapping.
    $$(H_1 + H_n, q) => (a)$$
    Where H is of `batch x n x dim` and q is of `batch x dim`.

    References:
    https://github.com/OpenNMT/OpenNMT-py/tree/fc23dfef1ba2f258858b2765d24565266526dc76/onmt/modules
    http://www.aclweb.org/anthology/D15-1166
    """

    def __init__(self, idf, cdf,  gamma1=4.0):
        super(GlobalAttentionGeneral, self).__init__()
        self.conv_context = conv1x1(cdf, idf)
        self.gamma1 = gamma1
        self.mask = None

    def applyMask(self, mask):
        # mask padding symbols (0)
        self.mask = mask  # B x T

    def forward(self, input, context):
        """
            input: B x idf x H x W (SR=H*W)
            context: B x cdf x T
            idf - image feature dimensionality
            cdf - text feature dimensionalty
            T - (max) length of caption in the batch
        """
        # H, W - size of the feature map of the image
        H, W = input.size(2), input.size(3)
        # SR - number of sub-regions
        SR = H * W
        # B - batch size
        B = context.size(0)

        # --> B x SR x idf
        target = input.view(B, -1, SR)
        targetT = torch.transpose(target, 1, 2).contiguous()
        # B x cdf x T --> B x cdf x T x 1
        sourceT = context.unsqueeze(3)
        # --> B x idf x T
        sourceT = self.conv_context(sourceT).squeeze(3)
        # Get attention
        # (B x SR x idf)(B x idf x T)
        # --> B x SR x T
        attn = torch.bmm(targetT, sourceT)
        if self.mask is not None:
            # B x T --> B x SR x T
            mask = self.mask.unsqueeze(1).repeat(1, SR, 1)
            # attn.data.masked_fill_(mask.data, -float('inf'))

        attn = torch.softmax(attn, dim=2) # Eq. (2)
        # --> B x T x SR
        attn = torch.transpose(attn, 1, 2).contiguous()

        # (B x idf x T)(B x T x SR)
        # --> B x idf x SR
        weightedContext = torch.bmm(sourceT, attn)
        weightedContext = weightedContext.view(B, -1, H, W)
        attn = attn.view(B, -1, H, W)

        return weightedContext, attn

