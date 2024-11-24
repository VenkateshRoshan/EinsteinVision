
import numpy as np
import sys

sys.path.append('../RAFT/')
sys.path.append('../RAFT/core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from core.utils import flow_viz
from core.utils.utils import InputPadder
import matplotlib.pyplot as plt


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.raft import RAFT
from core.update import BasicUpdateBlock, SmallUpdateBlock
from extractor import BasicEncoder, SmallEncoder
from corr import CorrBlock, AlternateCorrBlock
from core.utils.utils import bilinear_sampler, coords_grid, upflow8

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

# class RAFT(nn.Module):
#     def __init__(self, args):
#         super(RAFT, self).__init__()
#         self.args = args

#         if args.small:
#             self.hidden_dim = hdim = 96
#             self.context_dim = cdim = 64
#             args.corr_levels = 4
#             args.corr_radius = 3
        
#         else:
#             self.hidden_dim = hdim = 128
#             self.context_dim = cdim = 128
#             args.corr_levels = 4
#             args.corr_radius = 4

#         if 'dropout' not in self.args:
#             self.args.dropout = 0

#         if 'alternate_corr' not in self.args:
#             self.args.alternate_corr = False

#         # feature network, context network, and update block
#         if args.small:
#             self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)        
#             self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
#             self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

#         else:
#             self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)        
#             self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
#             self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

#     def freeze_bn(self):
#         for m in self.modules():
#             if isinstance(m, nn.BatchNorm2d):
#                 m.eval()

#     def initialize_flow(self, img):
#         """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
#         N, C, H, W = img.shape
#         coords0 = coords_grid(N, H//8, W//8, device=img.device)
#         coords1 = coords_grid(N, H//8, W//8, device=img.device)

#         # optical flow computed as difference: flow = coords1 - coords0
#         return coords0, coords1

#     def upsample_flow(self, flow, mask):
#         """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
#         N, _, H, W = flow.shape
#         mask = mask.view(N, 1, 9, 8, 8, H, W)
#         mask = torch.softmax(mask, dim=2)

#         up_flow = F.unfold(8 * flow, [3,3], padding=1)
#         up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

#         up_flow = torch.sum(mask * up_flow, dim=2)
#         up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
#         return up_flow.reshape(N, 2, 8*H, 8*W)


#     def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
#         """ Estimate optical flow between pair of frames """

#         image1 = 2 * (image1 / 255.0) - 1.0
#         image2 = 2 * (image2 / 255.0) - 1.0

#         image1 = image1.contiguous()
#         image2 = image2.contiguous()

#         hdim = self.hidden_dim
#         cdim = self.context_dim

#         # run the feature network
#         with autocast(enabled=self.args.mixed_precision):
#             fmap1, fmap2 = self.fnet([image1, image2])        
        
#         fmap1 = fmap1.float()
#         fmap2 = fmap2.float()
#         if self.args.alternate_corr:
#             corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
#         else:
#             corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

#         # run the context network
#         with autocast(enabled=self.args.mixed_precision):
#             cnet = self.cnet(image1)
#             net, inp = torch.split(cnet, [hdim, cdim], dim=1)
#             net = torch.tanh(net)
#             inp = torch.relu(inp)

#         coords0, coords1 = self.initialize_flow(image1)

#         if flow_init is not None:
#             coords1 = coords1 + flow_init

#         flow_predictions = []
#         for itr in range(iters):
#             coords1 = coords1.detach()
#             corr = corr_fn(coords1) # index correlation volume

#             flow = coords1 - coords0
#             with autocast(enabled=self.args.mixed_precision):
#                 net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

#             # F(t+1) = F(t) + \Delta(t)
#             coords1 = coords1 + delta_flow

#             # upsample predictions
#             if up_mask is None:
#                 flow_up = upflow8(coords1 - coords0)
#             else:
#                 flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            
#             flow_predictions.append(flow_up)

#         if test_mode:
#             return coords1 - coords0, flow_up
            
#         return flow_predictions

# DEVICE = 'cpu'
DEVICE = 'cuda'

def sampson_distance(F, p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    F = np.array(F)
    return np.abs(p2.T @ F @ p1) / (np.sqrt(F @ p1 @ p1.T + F.T @ p2 @ p2.T))

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def viz(img, flo,show=True):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)
    
    if show:
        plt.imshow(img_flo / 255.0)
        plt.show()
    
def raftInfer(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            viz(image1, flow_up)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='../RAFT/models/raft-kitti.pth' ,help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    raftInfer(args)