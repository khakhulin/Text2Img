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

    def __init__(self, idf, cdf,  gamma1=4.0, dim=1):
        super(GlobalAttentionGeneral, self).__init__()
        self.conv_context = conv1x1(cdf, idf)
        self.sm = nn.Softmax(dim=dim)
        self.gamma1 = gamma1
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask  # batch x sourceL

    def forward(self, query, context):
        """
           query: B x T x D
           context: B x H x W x D
        """
        B, D = query.size(0), query.size(2)
        H, W = context.size(1), context.size(2)
        SR = H * W

        # --> B x SR x D
        context = context.view(B, SR, D)
        # Get attention
        # (B x SR x D)(B x D x T)
        # --> B x SR x T
        attn = torch.bmm(context, query.transpose(1, 2))  # Eq. (7) in AttnGAN paper
        attn = torch.softmax(attn, dim=2)  # Eq. (8)

        attn = attn * self.gamma1
        attn = torch.softmax(attn, dim=1)  # Eq. (9)

        # (B x D x SR)(B x SR x T)
        # --> B x D x T
        weightedContext = torch.bmm(context.transpose(1, 2), attn)

        return weightedContext, attn.view(B, -1, H, W)

