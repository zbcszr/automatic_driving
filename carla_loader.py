import argparse
import os
import random
import time
import datetime
import math
import logging

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from tensorboardX import SummaryWriter

from helper import AverageMeter, save_checkpoint

# from torch.nn import functional as F


class CarlaNet(nn.Module):
    def __init__(self, dropout_vec=None):
        super(CarlaNet, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            # nn.Dropout(self.dropout_vec[0]),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            # nn.Dropout(self.dropout_vec[1]),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            # nn.Dropout(self.dropout_vec[2]),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            # nn.Dropout(self.dropout_vec[3]),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            # nn.Dropout(self.dropout_vec[4]),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            # nn.Dropout(self.dropout_vec[5]),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            # nn.Dropout(self.dropout_vec[6]),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            # nn.Dropout(self.dropout_vec[7]),
            nn.ReLU(),
        )

        self.img_fc = nn.Sequential(
                nn.Linear(8192, 512),
                nn.Dropout(0.3),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.Dropout(0.3),
                nn.ReLU(),
            )

        self.speed_fc = nn.Sequential(
                nn.Linear(1, 128),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.Dropout(0.5),
                nn.ReLU(),
            )

        self.emb_fc = nn.Sequential(
                nn.Linear(512+128, 512),
                nn.Dropout(0.5),
                nn.ReLU(),
            )

        self.control_branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 256),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(256, 256),
                # nn.Dropout(self.dropout_vec[i*2+14]),
                nn.ReLU(),
                nn.Linear(256, 3),
            ) for i in range(4)
        ])

        self.speed_branch = nn.Sequential(
                nn.Linear(512, 256),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(256, 256),
                # nn.Dropout(self.dropout_vec[1]),
                nn.ReLU(),
                nn.Linear(256, 1),
            )

    def forward(self, img, speed):
        img = self.conv_block(img)
        img = img.view(-1, 8192)
        img = self.img_fc(img)

        speed = self.speed_fc(speed)
        emb = torch.cat([img, speed], dim=1)
        emb = self.emb_fc(emb)

        pred_control = torch.cat(
            [out(emb) for out in self.control_branches], dim=1)
        pred_speed = self.speed_branch(img)
        return pred_control, pred_speed, img, emb


class UncertainNet(nn.Module):
    def __init__(self, structure=2, dropout_vec=None):
        super(UncertainNet, self).__init__()
        self.structure = structure

        if (self.structure < 2 or self.structure > 3):
            raise("Structure must be one of 2|3")

        self.uncert_speed_branch = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )
        if self.structure == 2:
            self.uncert_control_branches = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Linear(256, 3),
                ) for i in range(4)
            ])

        if self.structure == 3:
            self.uncert_control_branches = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Linear(256, 3),
            )

    def forward(self, img_emb, emb):
        if self.structure == 2:
            log_var_control = torch.cat(
                [un(emb) for un in self.uncert_control_branches], dim=1)
        if self.structure == 3:
            log_var_control = self.uncert_control_branches(emb)
            log_var_control = torch.cat([log_var_control for _ in range(4)],
                                        dim=1)

        log_var_speed = self.uncert_speed_branch(img_emb)

        return log_var_control, log_var_speed


class FinalNet(nn.Module):
    def __init__(self, structure=2, dropout_vec=None):
        super(FinalNet, self).__init__()
        self.structure = structure

        self.carla_net = CarlaNet(dropout_vec=dropout_vec)
        self.uncertain_net = UncertainNet(structure)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(
                    m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, img, speed):
        pred_control, pred_speed, img_emb, emb = self.carla_net(img, speed)
        log_var_control, log_var_speed = self.uncertain_net(img_emb, emb)

        return pred_control, pred_speed, log_var_control, log_var_speed


import glob

import numpy as np
import h5py
from torchvision import transforms
from torch.utils.data import Dataset

from imgaug import augmenters as iaa
from helper import RandomTransWrapper


class CarlaH5Data():
    def __init__(self,
                 train_folder,
                 eval_folder,
                 batch_size=4, num_workers=4, distributed=False):

        self.loaders = {
            "train": torch.utils.data.DataLoader(
                CarlaH5Dataset(
                    data_dir=train_folder,
                    train_eval_flag="train"),
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                shuffle=True
            ),
            "eval": torch.utils.data.DataLoader(
                CarlaH5Dataset(
                    data_dir=eval_folder,
                    train_eval_flag="eval"),
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                shuffle=False
            )}


class CarlaH5Dataset(Dataset):
    def __init__(self, data_dir,
                 train_eval_flag="train", sequence_len=200):
        self.data_dir = data_dir
        self.data_list = glob.glob(data_dir+'*.h5')
        self.data_list.sort()
        self.sequnece_len = sequence_len
        self.train_eval_flag = train_eval_flag

        self.build_transform()

    def build_transform(self):
        if self.train_eval_flag == "train":
            self.transform = transforms.Compose([
                transforms.RandomOrder([
                    RandomTransWrapper(
                        seq=iaa.GaussianBlur(
                            (0, 1.5)),
                        p=0.09),
                    RandomTransWrapper(
                        seq=iaa.AdditiveGaussianNoise(
                            loc=0,
                            scale=(0.0, 0.05),
                            per_channel=0.5),
                        p=0.09),
                    RandomTransWrapper(
                        seq=iaa.Dropout(
                            (0.0, 0.10),
                            per_channel=0.5),
                        p=0.3),
                    RandomTransWrapper(
                        seq=iaa.CoarseDropout(
                            (0.0, 0.10),
                            size_percent=(0.08, 0.2),
                            per_channel=0.5),
                        p=0.3),
                    RandomTransWrapper(
                        seq=iaa.Add(
                            (-20, 20),
                            per_channel=0.5),
                        p=0.3),
                    RandomTransWrapper(
                        seq=iaa.Multiply(
                            (0.9, 1.1),
                            per_channel=0.2),
                        p=0.4),
                    RandomTransWrapper(
                        seq=iaa.ContrastNormalization(
                            (0.8, 1.2),
                            per_channel=0.5),
                        p=0.09),
                    ]),
                transforms.ToTensor()])
        else:
            self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    ])

    def __len__(self):
        return self.sequnece_len * len(self.data_list)

    def __getitem__(self, idx):
        data_idx = idx // self.sequnece_len
        file_idx = idx % self.sequnece_len
        file_name = self.data_list[data_idx]

        with h5py.File(file_name, 'r') as h5_file:
            img = np.array(h5_file['rgb'])[file_idx]
            img = self.transform(img)
            target = np.array(h5_file['targets'])[file_idx]
            target = target.astype(np.float32)
            # 2 Follow lane, 3 Left, 4 Right, 5 Straight
            # -> 0 Follow lane, 1 Left, 2 Right, 3 Straight
            command = int(target[24])-2
            # Steer, Gas, Brake (0,1, focus on steer loss)
            target_vec = np.zeros((4, 3), dtype=np.float32)
            target_vec[command, :] = target[:3]
            # in km/h, <90
            speed = np.array([target[10]/90, ]).astype(np.float32)
            mask_vec = np.zeros((4, 3), dtype=np.float32)
            mask_vec[command, :] = 1

            # TODO
            # add preprocess
        return img, speed, target_vec.reshape(-1), \
            mask_vec.reshape(-1),

parser = argparse.ArgumentParser(description='Carla CIL training')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--speed-weight', default=1, type=float,
                    help='speed weight (default 0.1)')
parser.add_argument('--branch-weight', default=1, type=float,
                    help='branch weight')
parser.add_argument('--id', default="carla-1", type=str)
parser.add_argument('--train-dir',
                    default="/home/shifa/dataset/SeqTrain/",
                    type=str, metavar='PATH',
                    help='training dataset')
parser.add_argument('--eval-dir',
                    default="/home/shifa/dataset/SeqVal/",
                    type=str, metavar='PATH',
                    help='evaluation dataset')
parser.add_argument('--epochs', default=9, type=int, metavar='N',
                    help='number of total epochs to run (default: 90)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=10, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr-step', default=10, type=int,
                    help='learning rate step size')
parser.add_argument('--lr-gamma', default=0.5, type=float,
                    help='learning rate gamma')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=1000, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--evaluate-log', default="",
                    type=str, metavar='PATH',
                    help='path to log evaluation results (default: none)')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--net-structure', default=2, type=int,
                    help='Network structure 1|2|3|4.')
                    #  1 pure regression
                    #  2 uncertainty separate branch
                    #  3 uncertainty unify
                    #  4 uncertainty under branch


def output_log(output_str, logger=None):
    """
    standard output and logging
    """
    print("[{}]: {}".format(datetime.datetime.now(), output_str))
    if logger is not None:
        logger.critical("[{}]: {}".format(datetime.datetime.now(), output_str))


def log_args(logger):
    '''
    log args
    '''
    attrs = [(p, getattr(args, p)) for p in dir(args) if not p.startswith('_')]
    for key, value in attrs:
        output_log("{}: {}".format(key, value), logger=logger)


def main_run(arg):
    global args
    args = parser.parse_args(arg)
    log_dir = os.path.join("./", "logs", args.id)
    run_dir = os.path.join("./", "runs", args.id)
    save_weight_dir = os.path.join("./save_models", args.id)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_weight_dir, exist_ok=True)

    logging.basicConfig(filename=os.path.join(log_dir, "carla_training.log"),
                        level=logging.ERROR)
    tsbd = SummaryWriter(log_dir=run_dir)
    log_args(logging)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        output_log(
            'You have chosen to seed training. '
            'This will turn on the CUDNN deterministic setting, '
            'which can slow down your training considerably! '
            'You may see unexpected behavior when restarting '
            'from checkpoints.', logger=logging)

    if args.gpu is not None:
        output_log('You have chosen a specific GPU. This will completely '
                   'disable data parallelism.', logger=logging)

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url,
                                world_size=args.world_size,
                                rank=0)

    model = FinalNet(args.net_structure)
    criterion = nn.MSELoss()

    try:
        model.carla_net.load_state_dict(
            torch.load("./save_models/new_structure_best.pth")['state_dict'])
    except Exception as ee:
        print(ee)

    tsbd.add_graph(model,
                   (torch.zeros(1, 3, 88, 200),
                    torch.zeros(1, 1)))

    if args.gpu is not None:
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    optimizer = optim.Adam(
        model.uncertain_net.parameters(), args.lr, betas=(0.7, 0.85))
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    best_prec = math.inf

    # optionally resume from a checkpoint
    if args.resume:
        args.resume = os.path.join(save_weight_dir, args.resume)
        if os.path.isfile(args.resume):
            output_log("=> loading checkpoint '{}'".format(args.resume),
                       logging)
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['scheduler'])
            best_prec = checkpoint['best_prec']
            output_log("=> loaded checkpoint '{}' (epoch {})"
                       .format(args.resume, checkpoint['epoch']), logging)
        else:
            output_log("=> no checkpoint found at '{}'".format(args.resume),
                       logging)

    cudnn.benchmark = True

    carla_data = CarlaH5Data(
        train_folder=args.train_dir,
        eval_folder=args.eval_dir,
        batch_size=args.batch_size,
        num_workers=args.workers)

    train_loader = carla_data.loaders["train"]
    eval_loader = carla_data.loaders["eval"]

    if args.evaluate:
        args.id = args.id+"_test"
        if not os.path.isfile(args.resume):
            output_log("=> no checkpoint found at '{}'"
                       .format(args.resume), logging)
            return
        if args.evaluate_log == "":
            output_log("=> please set evaluate log path with --evaluate-log <log-path>")

        # evaluate(eval_loader, model, criterion, 0, tsbd)
        evaluate_test(model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        lr_scheduler.step()
        branch_losses, speed_losses, losses = \
            train(train_loader, model, criterion, optimizer, epoch, tsbd)

        prec = evaluate(eval_loader, model, criterion, epoch, tsbd)

        # remember best prec@1 and save checkpoint
        is_best = prec < best_prec
        best_prec = min(prec, best_prec)
        save_checkpoint(
            {'epoch': epoch + 1,
             'state_dict': model.state_dict(),
             'best_prec': best_prec,
             'scheduler': lr_scheduler.state_dict(),
             'optimizer': optimizer.state_dict()},
            args.id,
            is_best,
            os.path.join(
                save_weight_dir,
                "{}_{}.pth".format(epoch+1, args.id))
            )


def train(loader, model, criterion, optimizer, epoch, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    uncertain_losses = AverageMeter()
    ori_losses = AverageMeter()
    branch_losses = AverageMeter()
    speed_losses = AverageMeter()
    uncertain_control_means = AverageMeter()
    uncertain_speed_means = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    step = epoch * len(loader)
    for i, (img, speed, target, mask) in enumerate(loader):
        data_time.update(time.time() - end)

        # if args.gpu is not None:
        img = img.cuda(args.gpu, non_blocking=True)
        speed = speed.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        mask = mask.cuda(args.gpu, non_blocking=True)

        if args.net_structure != 1:
            branches_out, pred_speed, log_var_control, log_var_speed = model(img,
                                                                             speed)

            branch_square = torch.pow((branches_out - target), 2)
            branch_loss = torch.mean((torch.exp(-log_var_control)
                                      * branch_square
                                      + log_var_control) * 0.5 * mask) * 4

            speed_square = torch.pow((pred_speed - speed), 2)
            speed_loss = torch.mean((torch.exp(-log_var_speed)
                                     * speed_square
                                     + log_var_speed) * 0.5)

            uncertain_loss = args.branch_weight*branch_loss+args.speed_weight*speed_loss
            with torch.no_grad():
                ori_loss = args.branch_weight * torch.mean(branch_square*mask*4) \
                        + args.speed_weight * torch.mean(speed_square)
                uncertain_control_mean = torch.mean(torch.exp(log_var_control) * mask * 4)
                uncertain_speed_mean = torch.mean(torch.exp(log_var_speed))

                ori_losses.update(ori_loss.item(), args.batch_size)
                uncertain_control_means.update(uncertain_control_mean.item(),
                                               args.batch_size)
                uncertain_speed_means.update(uncertain_speed_mean.item(),
                                             args.batch_size)

        else:
            branches_out, pred_speed = model(img, speed)
            branch_loss = criterion(branches_out * mask, target) * 4
            speed_loss = criterion(pred_speed, speed)
            uncertain_loss = args.branch_weight * branch_loss \
                + args.speed_weight * speed_loss

        uncertain_losses.update(uncertain_loss.item(), args.batch_size)
        branch_losses.update(branch_loss.item(), args.batch_size)
        speed_losses.update(speed_loss.item(), args.batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        model.zero_grad()
        uncertain_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 or i+1 == len(loader):
            writer.add_scalar('train/branch_loss', branch_losses.val, step+i)
            writer.add_scalar('train/speed_loss', speed_losses.val, step+i)
            writer.add_scalar('train/uncertain_loss', uncertain_losses.val, step+i)
            writer.add_scalar('train/ori_loss', ori_losses.val, step+i)
            writer.add_scalar('train/control_uncertain',
                              uncertain_control_means.val, step+i)
            writer.add_scalar('train/speed_uncertain',
                              uncertain_speed_means.val, step+i)
            output_log(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Branch loss {branch_loss.val:.3f} ({branch_loss.avg:.3f})\t'
                'Speed loss {speed_loss.val:.3f} ({speed_loss.avg:.3f})\t'
                'Uncertain Loss {uncertain_loss.val:.4f} ({uncertain_loss.avg:.4f})\t'
                'Ori Loss {ori_loss.val:.4f} ({ori_loss.avg:.4f})\t'
                'Control Uncertain {control_uncertain.val:.4f} ({control_uncertain.avg:.4f})\t'
                'Speed Uncertain {speed_uncertain.val:.4f} ({speed_uncertain.avg:.4f})\t'
                .format(
                    epoch+1, i, len(loader), batch_time=batch_time,
                    data_time=data_time,
                    branch_loss=branch_losses,
                    speed_loss=speed_losses,
                    uncertain_loss=uncertain_losses,
                    ori_loss=ori_losses,
                    control_uncertain=uncertain_control_means,
                    speed_uncertain=uncertain_speed_means
                    ), logging)

    return branch_losses.avg, speed_losses.avg, uncertain_losses.avg


def evaluate(loader, model, criterion, epoch, writer):
    batch_time = AverageMeter()
    uncertain_losses = AverageMeter()
    ori_losses = AverageMeter()
    uncertain_control_means = AverageMeter()
    uncertain_speed_means = AverageMeter()

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (img, speed, target, mask) in enumerate(loader):
            img = img.cuda(args.gpu, non_blocking=True)
            speed = speed.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            mask = mask.cuda(args.gpu, non_blocking=True)

            branches_out, pred_speed, log_var_control, log_var_speed = model(img, speed)

            mask_out = branches_out * mask
            ori_branch_loss = criterion(mask_out, target) * 4
            ori_speed_loss = criterion(pred_speed, speed)

            branch_loss = torch.mean((torch.exp(-log_var_control)
                                      * torch.pow((branches_out - target), 2)
                                      + log_var_control) * 0.5 * mask) * 4

            speed_loss = torch.mean((torch.exp(-log_var_speed)
                                     * torch.pow((pred_speed - speed), 2)
                                     + log_var_speed) * 0.5)

            uncertain_loss = args.branch_weight*branch_loss + \
                    args.speed_weight*speed_loss
            ori_loss = args.branch_weight*ori_branch_loss + \
                    args.speed_weight*ori_speed_loss

            uncertain_control_mean = torch.mean(torch.exp(log_var_control) * mask * 4)
            uncertain_speed_mean = torch.mean(torch.exp(log_var_speed))

            uncertain_losses.update(uncertain_loss.item(), args.batch_size)
            ori_losses.update(ori_loss.item(), args.batch_size)
            uncertain_control_means.update(uncertain_control_mean.item(),
                                           args.batch_size)
            uncertain_speed_means.update(uncertain_speed_mean.item(),
                                         args.batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        writer.add_scalar('eval/uncertain_loss', uncertain_losses.avg, epoch+1)
        writer.add_scalar('eval/origin_loss', ori_losses.avg, epoch+1)
        writer.add_scalar('eval/control_uncertain',
                          uncertain_control_means.avg, epoch+1)
        writer.add_scalar('eval/speed_uncertain',
                          uncertain_speed_means.avg, epoch+1)
        output_log(
          'Epoch Test: [{0}]\t'
          'Time {batch_time.avg:.3f}\t'
          'Uncertain Loss {uncertain_loss.avg:.4f}\t'
          'Original Loss {ori_loss.avg:.4f}\t'
          'Control Uncertain {control_uncertain.avg:.4f}\t'
          'Speed Uncertain {speed_uncertain.avg:.4f}\t'
          .format(
              epoch+1, batch_time=batch_time,
              uncertain_loss=uncertain_losses,
              ori_loss=ori_losses,
              control_uncertain=uncertain_control_means,
              speed_uncertain=uncertain_speed_means,
              ), logging)
    return uncertain_losses.avg

def evaluate_test(model, criterion):
    carla_data = CarlaH5Data(
        train_folder=args.train_dir,
        eval_folder='/home/shifa/dataset/test/',
        batch_size=1, # args.batch_size,
        num_workers=args.workers)

    loader = carla_data.loaders["eval"]
    print('len loader:', len(loader))
    
    batch_time = AverageMeter()
    uncertain_losses = AverageMeter()
    ori_losses = AverageMeter()
    uncertain_control_means = AverageMeter()
    uncertain_speed_means = AverageMeter()

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (img, speed, target, mask) in enumerate(loader):
            img = img.cuda(args.gpu, non_blocking=True)
            speed = speed.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            mask = mask.cuda(args.gpu, non_blocking=True)

            branches_out, pred_speed, log_var_control, log_var_speed = model(img, speed)

            mask_out = branches_out * mask
            #print('mask_out', mask_out[0], 'target', target[0], 'log_var_control', log_var_control[0])
            print('speed:', np.around(speed.cpu().detach().numpy(), 1), 'pred_speed:', np.around(pred_speed.cpu().detach().numpy(), 1))
            print('target:', np.around(target.cpu().detach().numpy(), 1), 'mask_out:', np.around(mask_out.cpu().detach().numpy(), 1))
            ori_branch_loss = criterion(mask_out, target) * 4
            ori_speed_loss = criterion(pred_speed, speed)
            #print('ori_branch_loss', ori_branch_loss, 'ori_speed_loss', ori_speed_loss)

            branch_loss = torch.mean((torch.exp(-log_var_control)
                                      * torch.pow((branches_out - target), 2)
                                      + log_var_control) * 0.5 * mask) * 4

            speed_loss = torch.mean((torch.exp(-log_var_speed)
                                     * torch.pow((pred_speed - speed), 2)
                                     + log_var_speed) * 0.5)
            print('branch_loss', branch_loss.shape, 'speed_loss', speed_loss.shape)

            uncertain_loss = args.branch_weight*branch_loss + \
                    args.speed_weight*speed_loss
            ori_loss = args.branch_weight*ori_branch_loss + \
                    args.speed_weight*ori_speed_loss

            uncertain_control_mean = torch.mean(torch.exp(log_var_control) * mask * 4)
            uncertain_speed_mean = torch.mean(torch.exp(log_var_speed))

            uncertain_losses.update(uncertain_loss.item(), args.batch_size)
            ori_losses.update(ori_loss.item(), args.batch_size)
            print('ori_losses', ori_losses.avg, 'ori_loss', ori_loss.shape)
            uncertain_control_means.update(uncertain_control_mean.item(),
                                           args.batch_size)
            uncertain_speed_means.update(uncertain_speed_mean.item(),
                                         args.batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        output_log(
          'Time {batch_time.avg:.3f}\t'
          'Uncertain Loss {uncertain_loss.avg:.4f}\t'
          'Original Loss {ori_loss.avg:.4f}\t'
          'Control Uncertain {control_uncertain.avg:.4f}\t'
          'Speed Uncertain {speed_uncertain.avg:.4f}\t'
          .format(
              batch_time=batch_time,
              uncertain_loss=uncertain_losses,
              ori_loss=ori_losses,
              control_uncertain=uncertain_control_means,
              speed_uncertain=uncertain_speed_means,
              ), logging)
    return uncertain_losses.avg

#     log_path = '/home/bixin/dlwpt-code/carla_cil_pytorch/logs/carla-1'
# main_run(['--resume', '9_carla-1.pth', '--evaluate', '--evaluate-log', log_path])