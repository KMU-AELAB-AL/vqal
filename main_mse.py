import os
import random

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler

from torchvision.datasets import CIFAR100, CIFAR10

from tqdm import tqdm

from config import *
from data.transform import Cifar

from models.resnet import ResNet18
from models.featurenet import FeatureNet

import autoencoder.models.ae as ae
import autoencoder.models.vae as vae

from data.sampler import SubsetSequentialSampler


random.seed('KMU_AELAB')
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True


transforms = Cifar()

if DATASET == 'cifar10':
    data_train = CIFAR10('./data', train=True, download=True, transform=transforms.train_transform)
    data_unlabeled = CIFAR10('./data', train=True, download=True, transform=transforms.test_transform)
    data_module = CIFAR10('./data', train=True, download=True, transform=transforms.test_transform)
    data_test = CIFAR10('./data', train=False, download=True, transform=transforms.test_transform)
elif DATASET == 'cifar100':
    data_train = CIFAR100('./data', train=True, download=True, transform=transforms.train_transform)
    data_unlabeled = CIFAR100('./data', train=True, download=True, transform=transforms.test_transform)
    data_module = CIFAR100('./data', train=True, download=True, transform=transforms.test_transform)
    data_test = CIFAR100('./data', train=False, download=True, transform=transforms.test_transform)


def train_module(models, optimizers, dataloaders):
    models['ae'].eval()
    models['backbone'].eval()
    models['module'].train()

    _loss, cnt = 0., 0
    for data in tqdm(dataloaders['module'], leave=False, total=len(dataloaders['module'])):
        cnt += 1

        inputs = data[0].cuda()

        optimizers['module'].zero_grad()

        _, features = models['backbone'](inputs)

        features[0] = features[0].detach()
        features[1] = features[1].detach()
        features[2] = features[2].detach()
        features[3] = features[3].detach()

        pred_feature = models['module'](features)
        pred_feature = pred_feature.view([-1, EMBEDDING_DIM])
        ae_out = models['ae'](inputs)

        loss = torch.mean(torch.mean((pred_feature - ae_out[1].detach()) ** 2, dim=1))

        loss.backward()
        optimizers['module'].step()

        _loss += loss

    return _loss / cnt


def train_epoch(models, criterion, optimizers, dataloaders):
    models['backbone'].train()
    models['module'].eval()
    models['ae'].eval()

    _weight = WEIGHT
    for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
        inputs = data[0].cuda()
        labels = data[1].cuda()

        optimizers['backbone'].zero_grad()

        scores, features = models['backbone'](inputs)
        loss = criterion(scores, labels)

        loss.backward()
        optimizers['backbone'].step()


def test(models, dataloaders, mode='val'):
    models['backbone'].eval()
    models['module'].eval()

    total = 0
    correct = 0
    with torch.no_grad():
        for (inputs, labels) in dataloaders[mode]:
            inputs = inputs.cuda()
            labels = labels.cuda()

            scores, _ = models['backbone'](inputs)
            _, preds = torch.max(scores.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return 100 * correct / total


def train(models, criterion, optimizers, schedulers, dataloaders, num_epochs):
    print('>> Train a Model.')

    for epoch in range(num_epochs):
        schedulers['backbone'].step()
        train_epoch(models, criterion, optimizers, dataloaders)

    for epoch in range(num_epochs * 2):
        loss = train_module(models, optimizers, dataloaders)
        if loss < 0.3:
            return
        schedulers['module'].step(loss)

    print('>> Finished.')


def get_uncertainty(models, unlabeled_loader):
    models['backbone'].eval()
    models['module'].eval()
    models['ae'].eval()

    uncertainty = torch.tensor([]).cuda()
    with torch.no_grad():
        for (inputs, labels) in unlabeled_loader:
            inputs = inputs.cuda()

            _, features = models['backbone'](inputs)
            pred_feature = models['module'](features)
            pred_feature = pred_feature.view([-1, EMBEDDING_DIM])

            ae_out = models['ae'](inputs)

            loss = torch.mean((pred_feature - ae_out[1].detach()) ** 2, dim=1)

            uncertainty = torch.cat((uncertainty, loss), 0)

    return uncertainty.cpu()


if __name__ == '__main__':
    target_module = vae.VAE(NUM_RESIDUAL_LAYERS, NUM_RESIDUAL_HIDDENS, EMBEDDING_DIM)
    checkpoint = torch.load(f'trained_ae/vae_{DATASET}.pth.tar')
    target_module.load_state_dict(checkpoint['ae_state_dict'])
    target_module.cuda()

    for trial in range(TRIALS):
        fp = open(f'record_{trial + 1}.txt', 'w')

        indices = list(range(NUM_TRAIN))
        random.shuffle(indices)
        labeled_set = indices[:INIT_CNT]
        unlabeled_set = indices[INIT_CNT:]

        train_loader = DataLoader(data_train, batch_size=BATCH,
                                  sampler=SubsetRandomSampler(labeled_set),
                                  pin_memory=True)
        test_loader = DataLoader(data_test, batch_size=BATCH)
        module_train_loader = DataLoader(data_module, batch_size=BATCH,
                                         sampler=SubsetRandomSampler(labeled_set),
                                         pin_memory=True)
        dataloaders = {'train': train_loader, 'test': test_loader, 'module': module_train_loader}

        resnet18 = ResNet18(num_classes=CLS_CNT).cuda()
        feature_module = FeatureNet(out_dim=EMBEDDING_DIM).cuda()

        models = {'backbone': resnet18, 'module': feature_module, 'ae': target_module}

        torch.backends.cudnn.benchmark = False

        for cycle in range(CYCLES):
            criterion = nn.CrossEntropyLoss().cuda()
            optim_backbone = optim.SGD(models['backbone'].parameters(), lr=LR,
                                       momentum=MOMENTUM, weight_decay=WDECAY)
            optim_module = optim.Adam(models['module'].parameters(), lr=1e-3)

            sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)
            sched_module = lr_scheduler.ReduceLROnPlateau(optim_module, mode='min', factor=0.8, cooldown=4)

            optimizers = {'backbone': optim_backbone, 'module': optim_module}
            schedulers = {'backbone': sched_backbone, 'module': sched_module}

            train(models, criterion, optimizers, schedulers, dataloaders, EPOCH)
            acc = test(models, dataloaders, mode='test')

            fp.write(f'{acc}\n')
            print('Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {}'.format(trial + 1, TRIALS, cycle + 1,
                                                                                        CYCLES, len(labeled_set), acc))

            random.shuffle(unlabeled_set)
            subset = unlabeled_set[:]

            unlabeled_loader = DataLoader(data_unlabeled, batch_size=BATCH,
                                          sampler=SubsetSequentialSampler(subset),
                                          pin_memory=True)

            uncertainty = get_uncertainty(models, unlabeled_loader)

            arg = np.argsort(uncertainty)

            labeled_set += list(torch.tensor(subset)[arg][-ADDENDUM:].numpy())
            unlabeled_set = list(torch.tensor(subset)[arg][:-ADDENDUM].numpy()) + unlabeled_set[SUBSET:]

            dataloaders['train'] = DataLoader(data_train, batch_size=BATCH,
                                              sampler=SubsetRandomSampler(labeled_set),
                                              pin_memory=True)
