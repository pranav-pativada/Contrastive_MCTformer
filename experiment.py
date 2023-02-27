import numpy as np
from operator import itemgetter
import os
import code

import math
import sys
import torch

if __name__ == "__main__":

    criterion = None

    # randomise dim 
    b, n, c, d = 64, 196, 20, 384

    patches = torch.rand((b, c, n)).permute((0, 2, 1)).flatten(start_dim=0, end_dim=1).unsqueeze(dim=2)
    features = torch.rand((b, n, d)).flatten(start_dim=0, end_dim=1).unsqueeze(dim=1)

    fg_feats = torch.matmul(patches, features)
    bg_feats = torch.matmul(1-patches, features)
    clr_loss = 0
    for i in range(c):
        ith_fg_feat = fg_feats[:, i:, ]
        ith_bg_feat = bg_feats[:, i:, ]
        ith_clr_loss = criterion[0](ith_fg_feat) + criterion[1](ith_fg_feat, ith_bg_feat) + criterion[2](ith_bg_feat)
        clr_loss += ith_clr_loss

    #code.interact(local=dict(globals(), **locals()))
