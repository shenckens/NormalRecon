import argparse
import os
import time
import datetime

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from loguru import logger

from utils import tensor2float, save_scalars, DictAverageMeter, SaveScene, make_nograd_func
from datasets import transforms, find_dataset_def
from datasets.scannet import ScanNetDataset
from models import NeuralRecon
from config import cfg, update_config
from datasets.sampler import DistributedSampler
from ops.comm import *


def args():
    parser = argparse.ArgumentParser(description='A PyTorch Implementation of NeuralRecon')

    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # distributed training
    parser.add_argument('--gpu',
                        help='gpu id for multiprocessing training',
                        type=str)
    parser.add_argument('--world-size',
                        default=1,
                        type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--dist-url',
                        default='tcp://127.0.0.1:23456',
                        type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--local_rank',
                        default=0,
                        type=int,
                        help='node rank for distributed training')
    parser.add_argument('--normal_prior',
                        default=False,
                        type=bool,
                        help='Switch whether or not to include normal priors.')
    parser.add_argument('--prior_through_backbone',
                        default=False,
                        type=bool,
                        help='Puts normal imgs through the feature backbone model.')

    # parse arguments and check
    args = parser.parse_args()

    return args


def nnet_args():
    parser = argparse.ArgumentParser(description='Arguments for the normal prior model.')

    parser.add_argument('--architecture', default='BN', type=str, help='{BN, GN}')
    parser.add_argument("--pretrained", default='scannet', type=str, help="{nyu, scannet}")
    parser.add_argument('--sampling_ratio', type=float, default=0.4)
    parser.add_argument('--importance_ratio', type=float, default=0.7)
    parser.add_argument('--input_height', default=480, type=int)
    parser.add_argument('--input_width', default=640, type=int)
    # parser.add_argument('--imgs_dir', default='./examples', type=str)

    args, _ = parser.parse_known_args()

    return args


args = args()
update_config(cfg, args)

nnet_args = nnet_args() if args.normal_prior else False
prior_through_backbone = args.prior_through_backbone

cfg.defrost()
num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
print('number of gpus: {}'.format(num_gpus))
cfg.DISTRIBUTED = num_gpus > 1

if cfg.DISTRIBUTED:
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://"
    )
    synchronize()
cfg.LOCAL_RANK = args.local_rank
cfg.freeze()

torch.manual_seed(cfg.SEED)
torch.cuda.manual_seed(cfg.SEED)

# create logger
if is_main_process():
    if not os.path.isdir(cfg.LOGDIR):
        os.makedirs(cfg.LOGDIR)

    current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    logfile_path = os.path.join(cfg.LOGDIR, f'{current_time_str}_{cfg.MODE}.log')
    print('creating log file', logfile_path)
    logger.add(logfile_path, format="{time} {level} {message}", level="INFO")

    tb_writer = SummaryWriter(cfg.LOGDIR)

# Augmentation
if cfg.MODE == 'train':
    n_views = cfg.TRAIN.N_VIEWS
    random_rotation = cfg.TRAIN.RANDOM_ROTATION_3D
    random_translation = cfg.TRAIN.RANDOM_TRANSLATION_3D
    paddingXY = cfg.TRAIN.PAD_XY_3D
    paddingZ = cfg.TRAIN.PAD_Z_3D
else:
    n_views = cfg.TEST.N_VIEWS
    random_rotation = False
    random_translation = False
    paddingXY = 0
    paddingZ = 0

transform = []
transform += [transforms.ResizeImage((640, 480)),
              transforms.ToTensor(),
              transforms.RandomTransformSpace(
                  cfg.MODEL.N_VOX, cfg.MODEL.VOXEL_SIZE, random_rotation, random_translation,
                  paddingXY, paddingZ, max_epoch=cfg.TRAIN.EPOCHS),
              transforms.IntrinsicsPoseToProjection(n_views, 4),
              ]

transforms = transforms.Compose(transform)

# dataset, dataloader
MVSDataset = find_dataset_def(cfg.DATASET)
train_dataset = MVSDataset(cfg.TRAIN.PATH, "train", transforms, cfg.TRAIN.N_VIEWS, len(cfg.MODEL.THRESHOLDS) - 1)
# val_dataset = MVSDataseet(cfg.TRAIN.PATH, "val", transforms, cgf.TRAIN.N_VIEWS, len(cfg.MODEL.THRESHOLDS) - 1)
test_dataset = MVSDataset(cfg.TEST.PATH, "test", transforms, cfg.TEST.N_VIEWS, len(cfg.MODEL.THRESHOLDS) - 1)

# train_dataset = ScanNetDataset(cfg.TRAIN.PATH, "train", transforms, cfg.TRAIN.N_VIEWS, len(cfg.MODEL.THRESHOLDS) - 1, nnet_args)
# # val_dataset = MVSDataseet(cfg.TRAIN.PATH, "val", transforms, cgf.TRAIN.N_VIEWS, len(cfg.MODEL.THRESHOLDS) - 1, nnet_args)
# test_dataset = ScanNetDataset(cfg.TEST.PATH, "test", transforms, cfg.TEST.N_VIEWS, len(cfg.MODEL.THRESHOLDS) - 1, nnet_args)


if cfg.DISTRIBUTED:
    train_sampler = DistributedSampler(train_dataset, shuffle=False)
    TrainImgLoader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        sampler=train_sampler,
        num_workers=cfg.TRAIN.N_WORKERS,
        pin_memory=True,
        drop_last=True
    )
    test_sampler = DistributedSampler(test_dataset, shuffle=False)
    TestImgLoader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        sampler=test_sampler,
        num_workers=cfg.TEST.N_WORKERS,
        pin_memory=True,
        drop_last=False
    )
else:
    TrainImgLoader = DataLoader(train_dataset, cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.TRAIN.N_WORKERS,
                                drop_last=True)
    # ValImgLoader = DataLoader(val_dataset, cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.TRAIN.N_WORKERS,
    #                           drop_last=False)
    TestImgLoader = DataLoader(test_dataset, cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.TEST.N_WORKERS,
                               drop_last=False)

# model, optimizer
model = NeuralRecon(cfg, nnet_args=nnet_args, prior_through_backbone=prior_through_backbone)
if cfg.DISTRIBUTED:
    model.cuda()
    model = DistributedDataParallel(
        model, device_ids=[cfg.LOCAL_RANK], output_device=cfg.LOCAL_RANK,
        # this should be removed if we update BatchNorm stats
        broadcast_buffers=False,
        find_unused_parameters=True
    )
else:
    model = torch.nn.DataParallel(model, device_ids=[0])
    model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, betas=(0.9, 0.999), weight_decay=cfg.TRAIN.WD)


# main function
def train():
    # load parameters
    start_epoch = 0
    if cfg.RESUME:
        saved_models = [fn for fn in os.listdir(cfg.LOGDIR) if fn.endswith(".ckpt")]
        saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        if len(saved_models) != 0:
            # use the latest checkpoint file
            loadckpt = os.path.join(cfg.LOGDIR, saved_models[-1])
            logger.info("resuming " + str(loadckpt))
            map_location = {'cuda:%d' % 0: 'cuda:%d' % cfg.LOCAL_RANK}
            state_dict = torch.load(loadckpt, map_location=map_location)
            model.load_state_dict(state_dict['model'], strict=False)
            optimizer.param_groups[0]['initial_lr'] = state_dict['optimizer']['param_groups'][0]['lr']
            optimizer.param_groups[0]['lr'] = state_dict['optimizer']['param_groups'][0]['lr']
            start_epoch = state_dict['epoch'] + 1
    elif cfg.LOADCKPT != '':
        # load checkpoint file specified by args.loadckpt
        logger.info("loading model {}".format(cfg.LOADCKPT))
        map_location = {'cuda:%d' % 0: 'cuda:%d' % cfg.LOCAL_RANK}
        state_dict = torch.load(cfg.LOADCKPT, map_location=map_location)
        model.load_state_dict(state_dict['model'])
        optimizer.param_groups[0]['initial_lr'] = state_dict['optimizer']['param_groups'][0]['lr']
        optimizer.param_groups[0]['lr'] = state_dict['optimizer']['param_groups'][0]['lr']
        start_epoch = state_dict['epoch'] + 1
    logger.info("start at epoch {}".format(start_epoch))
    logger.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    milestones = [int(epoch_idx) for epoch_idx in cfg.TRAIN.LREPOCHS.split(':')[0].split(',')]
    lr_gamma = 1 / float(cfg.TRAIN.LREPOCHS.split(':')[1])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=lr_gamma,
                                                        last_epoch=start_epoch - 1)

    for epoch_idx in range(start_epoch, cfg.TRAIN.EPOCHS):
        logger.info('Epoch {}:'.format(epoch_idx))
        lr_scheduler.step()
        TrainImgLoader.dataset.epoch = epoch_idx
        TrainImgLoader.dataset.tsdf_cashe = {}
        # training
        for batch_idx, sample in enumerate(TrainImgLoader):
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % cfg.SUMMARY_FREQ == 0
            start_time = time.time()
            loss, scalar_outputs = train_sample(sample)
            if is_main_process():
                logger.info(
                    'Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx, cfg.TRAIN.EPOCHS,
                                                                                         batch_idx,
                                                                                         len(TrainImgLoader), loss,
                                                                                         time.time() - start_time))
            if do_summary and is_main_process():
                save_scalars(tb_writer, 'train', scalar_outputs, global_step)
            del scalar_outputs

        # checkpoint
        if (epoch_idx + 1) % cfg.SAVE_FREQ == 0 and is_main_process():
            torch.save({
                'epoch': epoch_idx,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()},
                "{}/model_{:0>6}.ckpt".format(cfg.LOGDIR, epoch_idx))


def test(from_latest=True):
    ckpt_list = []
    while True:
        saved_models = [fn for fn in os.listdir(cfg.LOGDIR) if fn.endswith(".ckpt")]
        saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))

        if from_latest:
            saved_models = saved_models[-1:]
        for ckpt in saved_models:
            if ckpt not in ckpt_list:
                # use the latest checkpoint file
                loadckpt = os.path.join(cfg.LOGDIR, ckpt)
                logger.info("resuming " + str(loadckpt))
                state_dict = torch.load(loadckpt)
                model.load_state_dict(state_dict['model'], strict=False)
                epoch_idx = state_dict['epoch']

                TestImgLoader.dataset.tsdf_cashe = {}

                avg_test_scalars = DictAverageMeter()
                save_mesh_scene = SaveScene(cfg)
                batch_len = len(TestImgLoader)
                for batch_idx, sample in enumerate(TestImgLoader):
                    for n in sample['fragment']:
                        logger.info(n)
                    # save mesh if SAVE_SCENE_MESH and is the last fragment
                    save_scene = cfg.SAVE_SCENE_MESH and batch_idx == batch_len - 1

                    start_time = time.time()
                    loss, scalar_outputs, outputs = test_sample(sample, save_scene)
                    logger.info('Epoch {}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(epoch_idx, batch_idx,
                                                                                                len(TestImgLoader),
                                                                                                loss,
                                                                                                time.time() - start_time))
                    avg_test_scalars.update(scalar_outputs)
                    del scalar_outputs

                    if batch_idx % 100 == 0:
                        logger.info("Iter {}/{}, test results = {}".format(batch_idx, len(TestImgLoader),
                                                                           avg_test_scalars.mean()))

                    # save mesh
                    if cfg.SAVE_SCENE_MESH:
                        save_mesh_scene(outputs, sample, epoch_idx)
                save_scalars(tb_writer, 'fulltest', avg_test_scalars.mean(), epoch_idx)
                logger.info("epoch {} avg_test_scalars:".format(epoch_idx), avg_test_scalars.mean())

                ckpt_list.append(ckpt)

        time.sleep(10)


def train_sample(sample):
    model.train()
    optimizer.zero_grad()

    outputs, loss_dict = model(sample)
    loss = loss_dict['total_loss']
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return tensor2float(loss), tensor2float(loss_dict)


@make_nograd_func
def test_sample(sample, save_scene=False):
    model.eval()

    outputs, loss_dict = model(sample, save_scene)
    loss = loss_dict['total_loss']

    return tensor2float(loss), tensor2float(loss_dict), outputs


if __name__ == '__main__':
    if cfg.MODE == "train":
        train()
    elif cfg.MODE == "test":
        test()
