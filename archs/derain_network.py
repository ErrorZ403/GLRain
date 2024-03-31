from archs.RLFN import RLFN
from archs.Restormer import DTB
import torch
import torch.nn as nn




class DerainNetwork(nn.Module):
    def __init__(self, dtb_config, rlfn_config):
        super().__init__()
        self.dtb = DTB(**dtb_config)
        self.rlfn = RLFN(**rlfn_config)
        
        embed_dim = dtb_config['dim']
        
        self.refiner = RLFN(
            in_channels=embed_dim,
            out_channels=3,
            feature_channels=embed_dim,
            num_blocks=2
        )
        
        self.conv = nn.Conv2d(embed_dim * 2, embed_dim, 1)
        
    def forward(self, x):
        x_dtb = self.dtb(x)
        x_rlfn = self.rlfn(x)
        
        x_concat = torch.cat([x_dtb, x_rlfn], dim=1)
        x_concat = self.conv(x_concat)
        
        x_refine = self.refiner(x_concat)
        #x_fused = self.fuser(x_rlfn)
        
        return x_refine