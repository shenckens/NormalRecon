import torch
import torch.nn as nn
import numpy as np
import os

from .backbone import MnasMulti
from .neucon_network import NeuConNet
from .gru_fusion import GRUFusion
from utils import tocuda
from models.NNet.NNET import NNET
import matplotlib.pyplot as plt


class NeuralRecon(nn.Module):
    '''
    NeuralRecon main class.
    '''

    def __init__(self, cfg, nnet_args=False):
        super(NeuralRecon, self).__init__()
        self.cfg = cfg.MODEL
        self.nnet_args = nnet_args
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
        self.neucon_net = NeuConNet(cfg.MODEL)
        # for fusing to global volume
        self.fuse_to_global = GRUFusion(cfg.MODEL, direct_substitute=True)
        self.one_time = True

    def normalizer(self, x):
        """ Normalizes the RGB images to the input range"""
        return (x - self.pixel_mean.type_as(x)) / self.pixel_std.type_as(x)

    def estm_norm_prior(self, img):
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        img = self.normalize(img)
        img.to(self.device)
        norm_out_list, _, _ = self.nnet(img)
        # includes norm and kappa (B, C, H, W) (1, 6, 480, 640)
        # remove Batchsize dimension
        norm_out = norm_out_list[-1].squeeze().detach()
        img.detach()
        return norm_out

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
        # makes 9 elements of B, C, H, W
        imgs = torch.unbind(inputs['imgs'], 1)

        # Normalize imgs beforehand.
        imgs = [self.normalizer(img) for img in imgs]

        # Add normal priors to images.
        if self.nnet_args:
            priors = []
            with torch.no_grad():
                for img in imgs:
                    print('imgshape', img.shape)
                    normal_list, _, _ = self.nnet(img)
                    normal_4c = normal_list[-1]
                    normal = normal_4c[:, :3, :, :]
                    kappa = normal_4c[:, 3:, :, :]
                    if self.one_time:
                        print("This is printed only once!")
                        # test = normal[0].permute(1, 2, 0).cpu()
                        print(test.shape)
                        plt.imsave('./normal_img.png', normal[0].permute(1, 2, 0).cpu().to_numpy())
                        plt.imsave('./kappa_img.png', kappa[0].permute(1, 2, 0).cpu().to_numpy())
                        self.one_time = False
                    print('normalshape', normal.shape)
                    # print(normals.shape)
                    prior = torch.cat([img, normal], dim=1)
                    priors.append(prior)
            imgs = priors

        # image feature extraction
        # in: images; out: feature maps
        # features = [self.backbone2d(self.normalizer(img)) for img in imgs] # 9 imgs

        # TODO: make it for for imgs.shape (bs, views, ch, h, w)
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
