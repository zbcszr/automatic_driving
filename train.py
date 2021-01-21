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

import h5py
import numpy as np

# print(torch.cuda.is_available())
# print(cudnn.version())

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
                nn.Linear(3, 128),
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

        self.control = nn.Sequential(
                nn.Linear(512, 256),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 3),
                # nn.LogSigmoid(),
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, img, motion):
        img = self.conv_block(img)
        img = img.view(-1, 8192)
        img = self.img_fc(img)

        motion = self.speed_fc(motion)
        emb = torch.cat([img, motion], dim=1)
        emb = self.emb_fc(emb)

        pred_motion = self.control(emb)

        return pred_motion

import glob

import numpy as np
import h5py
import torch
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
    # ContrastNormalization()` is deprecated. Use `imgaug.contrast.LinearContrast
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
                        seq=iaa.contrast.LinearContrast(
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
            # target bits: 0 forward, 1 backward, 2 left, 3 right, 4 keep, 5 speed, 6 turning, 7 go forward
            motion = target[:3]
            new_motion = target[3:]
        return img, motion, new_motion

def output_log(output_str, logger=None):
    """
    standard output and logging
    """
    print("[{}]: {}".format(datetime.datetime.now(), output_str))
    if logger is not None:
        logger.critical("[{}]: {}".format(datetime.datetime.now(), output_str))


def train(loader, model, criterion, optimizer, epoch, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    step = epoch * len(loader)
    for i, (img, motion, new_motion) in enumerate(loader):
        data_time.update(time.time() - end)

        # if args.gpu is not None:
        img = img.cuda(gpu, non_blocking=True)
        motion = motion.cuda(gpu, non_blocking=True)
        new_motion = new_motion.cuda(gpu, non_blocking=True)

        pred_motion = model(img, motion)

        loss = criterion(pred_motion, new_motion)

        losses.update(loss.item(), batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        # model.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0 or i == len(loader):
            writer.add_scalar('train/loss', losses.val, step+i)
            output_log(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                .format(
                    epoch, i, len(loader), batch_time=batch_time,
                    data_time=data_time, loss=losses), logging)

    return losses.avg

def evaluate(loader, model, criterion, epoch, writer):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (img, motion, new_motion) in enumerate(loader):
            img = img.cuda(gpu, non_blocking=True)
            motion = motion.cuda(gpu, non_blocking=True)
            new_motion = new_motion.cuda(gpu, non_blocking=True)

            pred_motion = model(img, motion)

            loss = criterion(pred_motion, new_motion)

            losses.update(loss.item(), batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        writer.add_scalar('eval/loss', losses.avg, epoch+1)

        output_log(
          'Epoch Test: [{0}]\t'
          'Time {batch_time.avg:.3f}\t'
          'Original Loss {ori_loss.avg:.4f}\t'
          .format(
              epoch+1, batch_time=batch_time,
              ori_loss=losses,
              ), logging)
    return losses.avg

logging.basicConfig(filename=os.path.join(log_dir, "carla_training.log"),
                        level=logging.ERROR)
tsbd = SummaryWriter(log_dir=run_dir)
model = CarlaNet()
criterion = nn.MSELoss()
tsbd.add_graph(model, (torch.zeros(1, 3, 88, 200), torch.zeros(1, 3)))

model = model.cuda(0)  # GPU id 0

optimizer = optim.Adam(
    model.parameters(), 0.001, betas=(0.7, 0.85))
lr_scheduler = optim.lr_scheduler.StepLR(
    optimizer, step_size=10, gamma=0.5)
best_prec = math.inf
cudnn.benchmark = True

carla_data = CarlaH5Data(
        train_folder='/home/shifa/dlwpt-code/data_storage/train/',
        eval_folder='/home/shifa/dlwpt-code/data_storage/eval/',
        batch_size=batch_size,
        num_workers=8)

train_loader = carla_data.loaders["train"]
eval_loader = carla_data.loaders["eval"]

for epoch in range(start_epoch, epochs):
        losses = train(train_loader, model, criterion, optimizer, epoch, tsbd)

        prec = evaluate(eval_loader, model, criterion, epoch, tsbd)

        lr_scheduler.step()

        # remember best prec@1 and save checkpoint
        is_best = prec < best_prec
        best_prec = min(prec, best_prec)
        save_checkpoint(
            {'epoch': epoch + 1,
             'state_dict': model.state_dict(),
             'best_prec': best_prec,
             'scheduler': lr_scheduler.state_dict(),
             'optimizer': optimizer.state_dict()},
            id,
            is_best,
            os.path.join(
                save_weight_dir,
                "{}_{}.pth".format(epoch+1, id))
            )

def load_model(resume):
    resume = os.path.join(save_weight_dir, resume)
    if os.path.isfile(resume):
        model = CarlaNet()
        model = model.cuda(0)  # GPU id 0

        output_log("=> loading checkpoint '{}'".format(resume), logging)
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['state_dict'])
        output_log("=> loaded checkpoint '{}' (epoch {})"
                       .format(resume, checkpoint['epoch']), logging)
        return model, checkpoint
    else:
        output_log("=> no checkpoint found at '{}'".format(resume), logging)

speed = 0
turn  = 0
forward = True
start = time.time()
model, checkpoint = load_model('9_training0117-LogSigmoid1.pth')
model.eval()
s.send(b"ONr 20 0")
while(1):
    ret,frame = cap.read()
    start = time.time()
    frame = cv2.resize(frame, (200,88))
    cv2.imshow('frame',frame)
    img = frame.astype(np.float32)
    img = torch.tensor([img])
    img = img.cuda(0, non_blocking=True)
    img = img.permute(0, 3, 1, 2).contiguous()
    old_motion = np.array([speed, forward, turn])
    motion = old_motion.astype(np.float32)
    motion = torch.tensor([motion])
    motion = motion.cuda(0, non_blocking=True)
    pred_motion = torch.round(model(img, motion))
    pred_motion = pred_motion.cpu().detach().numpy()
    end = time.time()
    k = cv2.waitKey(max(1, 30-int((end - start)*1000))) & 0xff

    if (old_motion != pred_motion).any():
        speed = int(pred_motion[0][0])
        forward = int(pred_motion[0][1])
        turn = int(pred_motion[0][2])
        cmd = "ONr {} {}".format(speed if forward else -speed, turn)
        print(cmd)
        s.send(cmd.encode("utf-8"))


    if k == 255:
        continue
    if k == 27:
        break

    if k == ord('t'):
        print(end - start)
        print(pred_motion)
        speed = int(pred_motion[0][0])
        forward = int(pred_motion[0][1])
        turn = int(pred_motion[0][2])
        if (old_motion != pred_motion).any():
            s.send("ONr {} {}".format(speed if forward else -speed, turn).encode("utf-8"))
    
    # s.send("ONr {} {}".format(speed if forward else -speed, turn).encode("utf-8"))
    
    if k == ord('l'):
        s.send(b"ONL")
    elif k == ord('r'):
        s.send(b"ONR")
    elif k == ord('u'):
        s.send(b"ONU")
    elif k == ord('d'):
        s.send(b"OND")
    elif k == ord('s'):
        s.send(b"ONs")

s.send(b"ONs")  # Stop running
cv2.destroyAllWindows()