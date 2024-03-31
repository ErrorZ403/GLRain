import argparse
from data.utils import load_img, save_img
import os
import numpy as np
import torch
from archs.derain_network import DerainNetwork

parser = argparse.ArgumentParser()
parser.add_argument("--img_dir" )
parser.add_argument("--save_dir")
parser.add_argument("--ckpt")

args = parser.parse_args()

os.makedirs(args['save_dir'], exist_ok=True)

derain_config = {
    'dtb_config': {
        'stride': 3,
        'num_blocks': [8],
        'dim': 48,
        'out_channels': 3
    },
    'nesr_config': {
        'feature_channels': 48,
        'out_channels': 48
    }
}
network = DerainNetwork(
    derain_config['dtb_config'],
    derain_config['nesr_config']
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
network.load_state_dict(torch.load(args['ckpt']))
network.to(device)

for img in os.listdir(os.listdir(args['img_dir'])):
    image = np.float32(load_img(f"{args['img_dir']}/{img}"))
    image = torch.from_numpy(image).permute(2, 0, 1)
    image = image.unsqueeze(0).to(device)

    restored = network(image)

    restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
    restored = restored * 255.

    save_img(f"{args['save_dir']}/{img}", restored)