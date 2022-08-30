import torch
import torch.nn as nn
import numpy as np
import os

from .backbone import MnasMulti
from .neucon_network import NeuConNet
from .gru_fusion import GRUFusion
from utils import tocuda
from models.NNet.NNET import NNET
from torchvision.utils import save_image
import torchvision.transforms as T


class NeuralRecon(nn.Module):
    '''
    NeuralRecon main class.
    '''

    def __init__(self, cfg, nnet_args=False, prior_through_backbone=False):
        super(NeuralRecon, self).__init__()
        self.cfg = cfg.MODEL
        self.nnet_args = nnet_args
        self.prior_through_backbone = prior_through_backbone
        alpha = float(self.cfg.BACKBONE2D.ARC.split('-')[-1])
        # other hparams
        self.pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1)
        self.n_scales = len(self.cfg.THRESHOLDS) - 1

        # networks
        self.backbone2d = MnasMulti(alpha)
        if self.nnet_args:
            self.nnet = NNET(nnet_args)
            loadckpt = os.path.join(cfg.TRAIN.PATH, 'scannet.pt')
            state_dict = torch.load(loadckpt)
            self.nnet.load_state_dict(state_dict['model'])
            # self.nnet.cuda()
            self.nnet.eval()
        self.neucon_net = NeuConNet(cfg.MODEL, nnet_args)
        # for fusing to global volume
        self.fuse_to_global = GRUFusion(cfg.MODEL, direct_substitute=True)
        self.one_time = True

    def normalizer(self, x):
        """ Normalizes the RGB images to the input range"""
        return (x - self.pixel_mean.type_as(x)) / self.pixel_std.type_as(x)


    def forward(self, inputs, save_mesh=False):
        '''

        :param inputs: dict: {
            'imgs':                    (Tensor), images,
                                    (batch size, number of views, C, H, W)
            'vol_origin':              (Tensor), origin of the full voxel volume (xyz position of voxel (0, 0, 0)),
                                    (batch size, 3)
            'vol_origin_partial':      (Tensor), origin of the partial voxel volume (xyz position of voxel (0, 0, 0)),
                                    (batch size, 3)
            'world_to_aligned_camera': (Tensor), matrices: transform from world coords to aligned camera coords,
                                    (batch size, number of views, 4, 4)
            'proj_matrices':           (Tensor), projection matrix,
                                    (batch size, number of views, number of scales, 4, 4)
            when we have ground truth:
            'tsdf_list':               (List), tsdf ground truth for each level,
                                    [(batch size, DIM_X, DIM_Y, DIM_Z)]
            'occ_list':                (List), occupancy ground truth for each level,
                                    [(batch size, DIM_X, DIM_Y, DIM_Z)]
            others: unused in network
        }
        :param save_mesh: a bool to indicate whether or not to save the reconstructed mesh of current sample
        :return: outputs: dict: {
            'coords':                  (Tensor), coordinates of voxels,
                                    (number of voxels, 4) (4 : batch ind, x, y, z)
            'tsdf':                    (Tensor), TSDF of voxels,
                                    (number of voxels, 1)
            When it comes to save results:
            'origin':                  (List), origin of the predicted partial volume,
                                    [3]
            'scene_tsdf':              (List), predicted tsdf volume,
                                    [(nx, ny, nz)]
        }
                 loss_dict: dict: {
            'tsdf_occ_loss_X':         (Tensor), multi level loss
            'total_loss':              (Tensor), total loss
        }
        '''
        inputs = tocuda(inputs)
        outputs = {}
        # Makes 9 elements of B, C, H, W.
        imgs = torch.unbind(inputs['imgs'], 1)

        # Normalize imgs beforehand.
        imgs = [self.normalizer(img) for img in imgs]

        # Add normal priors to images.
        if self.nnet_args:
            # image feature extraction
            # in: images; out: feature maps
            features = [self.backbone2d(img) for img in imgs]
            normals = []
            with torch.no_grad():
                for img in imgs:
                    normal_list, _, _ = self.nnet(img)
                    normal = normal_list[-1][:, :3, :, :]
                    normals.append(normal)
            # Normal rgb imgs through backbone feature extraction.
            if self.prior_through_backbone:
                normals_features = [self.backbone2d(normal) for normal in normals]
            # Normal imgs as features concatenated to original imgs' features.
            else:
                # Resize normal imgs to fit backbone feature outputs and concat.
                sizes = [(features[0][i].shape[2], features[0][i].shape[3]) for i in range(len(features[0]))]
                normals_features = [[T.Resize(size=size)(norm) for size in sizes] for norm in normals]

            concat_features = []
            for i in range(len(features)):
                elements = []
                for e in range(len(features[0])):
                    elements.append(torch.cat([features[i][e], normals_features[i][e]], dim=1))
                concat_features.append(elements)
            features = concat_features

        else:
            # image feature extraction
            # in: images; out: feature maps
            features = [self.backbone2d(img) for img in imgs]

        # coarse-to-fine decoder: SparseConv and GRU Fusion.
        # in: image feature; out: sparse coords and tsdf
        outputs, loss_dict = self.neucon_net(features, inputs, outputs)

        # fuse to global volume.
        if not self.training and 'coords' in outputs.keys():
            outputs = self.fuse_to_global(
                outputs['coords'], outputs['tsdf'], inputs, self.n_scales, outputs, save_mesh)

        # gather loss.
        print_loss = 'Loss: '
        for k, v in loss_dict.items():
            print_loss += f'{k}: {v} '

        weighted_loss = 0

        for i, (k, v) in enumerate(loss_dict.items()):
            weighted_loss += v * self.cfg.LW[i]

        loss_dict.update({'total_loss': weighted_loss})
        return outputs, loss_dict
