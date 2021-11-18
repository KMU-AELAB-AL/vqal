import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision.datasets import CIFAR100, CIFAR10

from tqdm import tqdm
from tensorboardX import SummaryWriter

from models.vae import VAE
from config import *
from transform import Cifar


random.seed('KMU_AELAB')
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True


transforms = Cifar()

if DATASET == 'cifar10':
    data_train = CIFAR10('../data', train=True, download=True, transform=transforms.transform)
    data_unlabeled = CIFAR10('../data', train=True, download=True, transform=transforms.transform)
    data_test = CIFAR10('../data', train=False, download=True, transform=transforms.transform)
elif DATASET == 'cifar100':
    data_train = CIFAR100('../data', train=True, download=True, transform=transforms.transform)
    data_unlabeled = CIFAR100('../data', train=True, download=True, transform=transforms.transform)
    data_test = CIFAR100('../data', train=False, download=True, transform=transforms.transform)


def train_epoch(model, criterion, opt, dataloaders, summary_writer, epoch):
    model.train()

    cnt = 0
    _loss = 0.
    for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
        cnt += 1

        opt.zero_grad()
        inputs = data[0].cuda()
        targets = data[1].cuda()

        recon, features, mu, logvar = model(inputs)
        recon_loss = criterion(recon, inputs)
        kld_loss = torch.sum(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)

        loss = recon_loss + kld_loss

        loss.backward()
        opt.step()

        _loss += loss

    summary_writer.add_image('image/origin', inputs[0], epoch)
    summary_writer.add_image('image/recon', recon[0], epoch)
    summary_writer.add_scalar('loss', _loss / cnt, epoch)

    if epoch % 100 == 99:
        summary_writer.add_embedding(features.detach(), metadata=targets.detach().cpu().numpy(),
                                     label_img=inputs.detach(), global_step=epoch)

    return _loss / cnt


def test(model, criterion, dataloaders, mode='val'):
    assert mode == 'val' or mode == 'test'
    model.eval()

    loss = 0.
    with torch.no_grad():
        for (inputs, _) in dataloaders[mode]:
            inputs = inputs.cuda()

            recon, _, _, _ = model(inputs)
            loss += criterion(recon, inputs)

    return loss


def train(model, criterion, opt, scheduler, dataloaders, num_epochs):
    print('>> Train a Model.')

    best_loss = 999999999.
    summary_writer = SummaryWriter(log_dir=os.path.join('./'), comment='VAE')
    if not os.path.exists('../trained_ae'):
        os.makedirs('../trained_ae')
    
    for epoch in range(num_epochs):
        loss = train_epoch(model, criterion, opt, dataloaders, summary_writer, epoch)
        scheduler.step(loss)

        if epoch % 5 == 4:
            _loss = test(model, criterion, dataloaders, 'test')
            if best_loss > _loss:
                best_loss = _loss
                torch.save(
                    {
                        'epoch': epoch + 1,
                        'ae_state_dict': model.state_dict(),
                    },
                    f'../trained_ae/vae_{DATASET}.pth.tar'
                )
            print('Val loss: {:.3f} \t Best loss: {:.3f}'.format(_loss, best_loss))
    print('>> Finished.')


if __name__ == '__main__':
    train_loader = DataLoader(data_train, batch_size=BATCH, pin_memory=True)
    test_loader = DataLoader(data_test, batch_size=BATCH)
    dataloaders = {'train': train_loader, 'test': test_loader}

    # Model
    model = VAE(NUM_RESIDUAL_LAYERS, NUM_RESIDUAL_HIDDENS, EMBEDDING_DIM).cuda()
    torch.backends.cudnn.benchmark = False

    criterion = nn.MSELoss(reduction='sum').cuda()
    opt = optim.Adam(model.parameters(), lr=LR)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.8, cooldown=4)

    train(model, criterion, opt, scheduler, dataloaders, EPOCH)
