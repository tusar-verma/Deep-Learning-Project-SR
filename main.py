from __future__ import print_function
import argparse
from math import log10
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Net
from data import train_test_split

import time
import dataDownloader


def train(epoch):
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        loss = criterion(model(input), target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.item()))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))


def test(epoch, save_dir):
    avg_psnr = 0
    with torch.no_grad():
        for batch in testing_data_loader:
            input, target = batch[0].to(device), batch[1].to(device)

            prediction = model(input)
            mse = criterion(prediction, target)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr

    avg_psnr_epoch = avg_psnr / len(testing_data_loader)
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr_epoch))

    with open(save_dir + "psnr_log.txt", "a") as f:
        f.write(f"Epoch {epoch}: {avg_psnr_epoch:.4f} dB\n")


def checkpoint(epoch, save_dir):
    model_out_path = save_dir + "checkpoint_epoch_{}.pth".format(epoch)
    torch.save({
        'epoch': epoch,
        'model': model,
        'optimizer': optimizer
    }, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


if __name__ ==  '__main__':
# Training settings
    parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
    parser.add_argument('--upscale_factor', type=int, required=True, help="super resolution upscale factor")
    parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
    parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
    parser.add_argument('--nEpochs', type=int, default=2, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
    parser.add_argument('--cuda', action='store_true', help='use cuda?')
    parser.add_argument('--mps', action='store_true', default=False, help='enables macOS GPU training')
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--datasetName', type=str, required=True, help='name of the dataset to use')
    parser.add_argument('--checkpointToContinueFrom', type=str, default='', help='name of checkpoint to continue training from')
    opt = parser.parse_args()

    print(opt)

    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    if not opt.mps and torch.backends.mps.is_available():
        raise Exception("Found mps device, please run with --mps to enable macOS GPU")

    torch.manual_seed(opt.seed)
    use_mps = opt.mps and torch.backends.mps.is_available()

    if opt.cuda:
        print("Using CUDA")
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print('===> Loading datasets')
  
    dataDownloader.downloadDataSet(opt.datasetName)
    train_set, test_set = train_test_split(source_path="./dataSets/"+ opt.datasetName, upscale_factor=opt.upscale_factor)
    
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

    # create directory for saving epochs for corresponding dataset
    epoch_save_dir = "./checkpoints/upscale_factor_x{}/{}/".format(opt.upscale_factor ,opt.datasetName)
    os.makedirs(os.path.dirname(epoch_save_dir), exist_ok=True)


    print('===> Building model')
    if opt.checkpointToContinueFrom:
        print("===> Loading model from checkpoint: {}".format(opt.checkpointToContinueFrom))
        aCheckpoint = torch.load(opt.checkpointToContinueFrom, weights_only=False)
        model = aCheckpoint['model']
        
        optimizer = aCheckpoint['optimizer']
        start_epoch = aCheckpoint['epoch'] + 1
    else:
        start_epoch = 1

        model = Net(upscale_factor=opt.upscale_factor).to(device)        
        optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    
    criterion = nn.MSELoss()

    for epoch in range(start_epoch, opt.nEpochs + 1):
        train(epoch)
        test(epoch, epoch_save_dir)
        checkpoint(epoch, epoch_save_dir)

    os.system("systemctl suspend -i")