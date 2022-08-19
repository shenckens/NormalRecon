import os
import numpy as np
import pickle
import cv2
import torch
from PIL import Image
from models.NNet.NNet import NNet
from torch.utils.data import Dataset
from torchvision import transforms

class ScanNetDataset(Dataset):
    def __init__(self, datapath, mode, transform, nviews, n_scales, nnet_args=False):
        super(ScanNetDataset, self).__init__()
        self.datapath = datapath
        self.mode = mode
        self.n_views = nviews
        self.transforms = transform
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.tsdf_file = 'all_tsdf_{}'.format(self.n_views)
        self.device = torch.device('cuda:0')
        if nnet_args:
            self.nnet = NNet(nnet_args)
            loadckpt = os.path.join(self.datapath, 'scannet.pt')
            state_dict = torch.load(loadckpt)
            self.nnet.load_state_dict(state_dict['model'])
            self.nnet.to(self.device)
            self.nnet.eval()

        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()
        if mode == 'test':
            self.source_path = 'scans_test'
        else:
            self.source_path = 'scans'

        self.n_scales = n_scales
        self.epoch = None
        self.tsdf_cashe = {}
        self.max_cashe = 100

    def build_list(self):
        with open(os.path.join(self.datapath, self.tsdf_file, 'fragments_{}.pkl'.format(self.mode)), 'rb') as f:
            metas = pickle.load(f)
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filepath, vid):
        intrinsics = np.loadtxt(os.path.join(filepath, 'intrinsic', 'intrinsic_color.txt'), delimiter=' ')[:3, :3]
        intrinsics = intrinsics.astype(np.float32)
        extrinsics = np.loadtxt(os.path.join(filepath, 'pose', '{}.txt'.format(str(vid))))
        return intrinsics, extrinsics

    def read_img(self, filepath):
        img = Image.open(filepath)
        return img

    def read_depth(self, filepath):
        # Read depth image and camera pose
        depth_im = cv2.imread(filepath, -1).astype(
            np.float32)
        depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
        depth_im[depth_im > 3.0] = 0
        return depth_im

    def estm_norm_prior(self, filepath):
        img = self.read_img(filepath)
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

    def read_scene_volumes(self, data_path, scene):
        if scene not in self.tsdf_cashe.keys():
            if len(self.tsdf_cashe) > self.max_cashe:
                self.tsdf_cashe = {}
            full_tsdf_list = []
            for l in range(self.n_scales + 1):
                # load full tsdf volume
                full_tsdf = np.load(os.path.join(data_path, scene, 'full_tsdf_layer{}.npz'.format(l)),
                                    allow_pickle=True)
                full_tsdf_list.append(full_tsdf.f.arr_0)
            self.tsdf_cashe[scene] = full_tsdf_list
        return self.tsdf_cashe[scene]

    def __getitem__(self, idx):
        meta = self.metas[idx]

        imgs = []
        depth = []
        normals = []
        extrinsics_list = []
        intrinsics_list = []

        tsdf_list = self.read_scene_volumes(os.path.join(self.datapath, self.tsdf_file), meta['scene'])

        for i, vid in enumerate(meta['image_ids']):
            # load images
            imgs.append(
                self.read_img(
                    os.path.join(self.datapath, self.source_path, meta['scene'], 'color', '{}.jpg'.format(vid)))
            )

            depth.append(
                self.read_depth(
                    os.path.join(self.datapath, self.source_path, meta['scene'], 'depth', '{}.png'.format(vid)))
            )

            normals.append(
                self.estm_norm_prior(
                    os.path.join(self.datapath, self.source_path, meta['scene'], 'color', '{}.jpg'.format(vid)))
            )

            # load intrinsics and extrinsics
            intrinsics, extrinsics = self.read_cam_file(os.path.join(self.datapath, self.source_path, meta['scene']),
                                                        vid)

            intrinsics_list.append(intrinsics)
            extrinsics_list.append(extrinsics)

        intrinsics = np.stack(intrinsics_list)
        extrinsics = np.stack(extrinsics_list)

        items = {
            'imgs': imgs,
            'depth': depth,
            'normals': normals,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'tsdf_list_full': tsdf_list,
            'vol_origin': meta['vol_origin'],
            'scene': meta['scene'],
            'fragment': meta['scene'] + '_' + str(meta['fragment_id']),
            'epoch': [self.epoch],
        }

        if self.transforms is not None:
            items = self.transforms(items)
        return items
