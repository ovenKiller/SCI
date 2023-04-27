import os
import sys
import time
import glob
import numpy as np
import torch
import utils
from PIL import Image
import logging
import argparse
from torch.utils.tensorboard import SummaryWriter
import torch.utils
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable
import pyiqa
from model import *
from multi_read_data import unpaired_data_loader
from multi_read_data import paired_data_loader
from datetime import datetime


dir = os.path.join('runs',datetime.now().strftime("%Y%m%d%H%M%S"))
writer = SummaryWriter(dir)

parser = argparse.ArgumentParser("SCI")
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--cuda', default=True, type=bool, help='Use CUDA to train model')
parser.add_argument('--gpu', type=str, default='0', help='gpu device id')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--epochs', type=int, default=300, help='epochs')
parser.add_argument('--lr', type=float, default=0.0003, help='learning rate')
parser.add_argument('--stage', type=int, default=3, help='epochs')
parser.add_argument('--save', type=str, default='EXP/', help='location of the data corpus')
parser.add_argument('--eval', type=str, default='data/eval', help='location of the data corpus')
parser.add_argument('--train', type=str, default='data/train', help='location of the data corpus')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

args.save = args.save + '/' + 'Train-{}'.format(time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
model_path = args.save + '/model_epochs/'
os.makedirs(model_path, exist_ok=True)
image_path = args.save + '/image_epochs/'
os.makedirs(image_path, exist_ok=True)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

logging.info("train file name = %s", os.path.split(__file__))

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


# def save_images(tensor, path):
#     image_numpy = tensor[0].cpu().float().numpy()
#     image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
#     im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
#     im.save(path, 'png')


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %s' % args.gpu)
    logging.info("args = %s", args)


    model = Network(stage=args.stage)

    model.enhance.in_conv.apply(model.weights_init)
    model.enhance.conv.apply(model.weights_init)
    model.enhance.out_conv.apply(model.weights_init)
    model.calibrate.in_conv.apply(model.weights_init)
    model.calibrate.convs.apply(model.weights_init)
    model.calibrate.out_conv.apply(model.weights_init)

    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=3e-4)
    MB = utils.count_parameters_in_MB(model)
    logging.info("model size = %f", MB)
    print(MB)


    train_low_data_names = 'Your train dataset'
    train_dataset = unpaired_data_loader(args.train)
    train_dataset.size = 256



    train_queue = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        pin_memory=True, num_workers=0, shuffle=True,generator=torch.Generator(device='cuda'))
    train_loader_len = len(train_queue)
    total_step = 0
    min_loss = 100000.0
    counter = 0
    psnr = pyiqa.create_metric('psnr').cuda()
    ssim = pyiqa.create_metric('ssim').cuda()
    lpips = pyiqa.create_metric('lpips').cuda()
    mae = torch.nn.L1Loss().cuda()
    eval_dataset = paired_data_loader(args.eval)
    eval_set_len = len(eval_dataset)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=8, shuffle=True, num_workers=0,generator=torch.Generator(device='cuda'))
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch_idx, low_light_img in enumerate(train_queue):
            total_step += 1
            low_light_img = low_light_img.cuda()

            optimizer.zero_grad()
            loss = model._loss(low_light_img)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(float(loss.item()))
            if(float(loss.item())<min_loss):
                min_loss = float(loss.item())
                utils.save(model, os.path.join(model_path, 'best.pt'))

            if batch_idx % ((train_loader_len-1)//10)==0:
                utils.save(model, os.path.join(model_path, 'latest.pt'))
                counter+=1
                test_model = Finetunemodel(model.state_dict())
                total_ssim = 0.0
                total_psnr = 0.0
                total_lpips = 0.0
                total_mae = 0.0
                writer.add_scalar("Loss",float(loss.item()),counter)
                with torch.no_grad():
                    for data,label in eval_loader:
                        data = data.cuda()
                        label = label.cuda()
                        _,enhanced_image = test_model(data)
                        total_ssim += float(ssim(enhanced_image,label).sum())
                        total_psnr += float(psnr(enhanced_image,label).sum())
                        total_lpips += float(lpips(enhanced_image,label).sum())
                        total_mae += float(mae(enhanced_image,label)*label.shape[0])
                
                writer.add_scalar("ssim",float(total_ssim/eval_set_len),counter)
                writer.add_scalar("psnr",float(total_psnr/eval_set_len),counter)
                writer.add_scalar("lpips",float(total_lpips/eval_set_len),counter)
                writer.add_scalar("mae",float(total_mae/eval_set_len),counter)
        logging.info('train-epoch %03d %f', epoch, np.average(losses))
        print(np.average(losses))

if __name__ == '__main__':
    main()
