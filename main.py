import os
import math
import random
from collections import Counter

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
from models.lossnet import LossNet

from autoencoder.models.vq_vae import VQVAE

from data.sampler import SubsetSequentialSampler


random.seed('KMU_AELAB')
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True


transforms = Cifar()

if DATASET == 'cifar10':
    data_train = CIFAR10('./data', train=True, download=True, transform=transforms.train_transform)
    data_unlabeled = CIFAR10('./data', train=True, download=True, transform=transforms.test_transform)
    data_test = CIFAR10('./data', train=False, download=True, transform=transforms.test_transform)
elif DATASET == 'cifar100':
    data_train = CIFAR100('./data', train=True, download=True, transform=transforms.train_transform)
    data_unlabeled = CIFAR100('./data', train=True, download=True, transform=transforms.test_transform)
    data_test = CIFAR100('./data', train=False, download=True, transform=transforms.test_transform)


def loss_pred_loss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape

    input = (input - input.flip(0))[:len(input) // 2]
    target = (target - target.flip(0))[:len(target) // 2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1

    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0)
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()

    return loss


def train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss):
    models['backbone'].train()
    models['module'].train()

    for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
        inputs = data[0].cuda()
        labels = data[1].cuda()

        optimizers['backbone'].zero_grad()
        optimizers['module'].zero_grad()

        scores, features = models['backbone'](inputs)
        target_loss = criterion(scores, labels)

        if epoch > epoch_loss:
            features[0] = features[0].detach()
            features[1] = features[1].detach()
            features[2] = features[2].detach()
            features[3] = features[3].detach()
        pred_loss = models['module'](features)
        pred_loss = pred_loss.view(pred_loss.size(0))

        m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
        m_module_loss = loss_pred_loss(pred_loss, target_loss, margin=MARGIN)
        loss = m_backbone_loss + WEIGHT * m_module_loss

        loss.backward()
        optimizers['backbone'].step()
        optimizers['module'].step()


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


def train(models, criterion, optimizers, schedulers, dataloaders, num_epochs, epoch_loss):
    print('>> Train a Model.')

    checkpoint_dir = os.path.join(f'./trained', 'weights')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    for epoch in range(num_epochs):
        schedulers['backbone'].step()
        schedulers['module'].step()

        train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss)

    print('>> Finished.')


def get_uncertainty(models, code_gen, index_idf, unlabeled_loader, train_loader, subset):
    models['backbone'].eval()
    models['module'].eval()

    labeled_indices = []
    for curr_it, data in enumerate(tqdm(train_loader, leave=False, total=len(train_loader))):
        inputs = data[0].cuda()

        _, _, _, indices = code_gen(inputs)
        indices = indices.cpu().tolist()

        for idx in range(len(indices)):
            labeled_indices += indices[idx]

    #############################
    labeled_index_cnt = Counter(labeled_indices)
    labeled_index_set = set(labeled_index_cnt.keys())


    index = 0
    unlabeled_set = []
    for curr_it, data in enumerate(tqdm(train_loader, leave=False, total=len(unlabeled_loader))):
        inputs = data[0].cuda()

        _, _, _, indices = code_gen(inputs)
        indices = indices.cpu().tolist()

        for idx in range(len(indices)):
            tmp_indices = set(indices[idx])
            unlabeled_set.append([subset[index],
                                  sum([index_idf[key] * labeled_index_cnt[key]
                                       for key in set(tmp_indices) & labeled_index_set])])
            index += 1

    ordered_set = list(np.array(sorted(unlabeled_set, key=lambda x: x[1]))[:, 0])

    return ordered_set


def set_idf(code_gen, dataloader):
    index_lst = []
    for curr_it, data in enumerate(tqdm(dataloader, leave=False, total=len(dataloader))):
        inputs = data[0].cuda()

        _, _, _, indices = code_gen(inputs)
        indices = indices.cpu().tolist()

        for idx in range(len(indices)):
            index_lst += indices[idx]

    index_cnt = Counter(index_lst)

    index_idf = {}
    for key in index_cnt:
        index_idf[key] = math.log(NUM_TRAIN / (1 + index_cnt[key]))

    return index_idf


if __name__ == '__main__':
    code_gen = VQVAE(NUM_HIDDENS, NUM_RESIDUAL_LAYERS, NUM_RESIDUAL_HIDDENS, NUM_EMBEDDINGS, EMBEDDING_DIM)
    checkpoint = torch.load(f'trained_ae/vqvae_{DATASET}.pth.tar')
    code_gen.load_state_dict(checkpoint['ae_state_dict'])
    code_gen.cuda()

    dataloader = DataLoader(data_unlabeled, batch_size=BATCH, pin_memory=True)
    index_idf = set_idf(code_gen, dataloader)

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
        dataloaders = {'train': train_loader, 'test': test_loader}

        loss_module = LossNet().cuda()
        resnet18 = ResNet18(num_classes=CLS_CNT).cuda()
        models = {'backbone': resnet18, 'module': loss_module}

        torch.backends.cudnn.benchmark = False

        for cycle in range(CYCLES):
            criterion = nn.CrossEntropyLoss(reduction='none').cuda()

            optim_backbone = optim.SGD(models['backbone'].parameters(), lr=LR,
                                       momentum=MOMENTUM, weight_decay=WDECAY)
            optim_module = optim.SGD(models['module'].parameters(), lr=LR,
                                     momentum=MOMENTUM, weight_decay=WDECAY)
            optimizers = {'backbone': optim_backbone, 'module': optim_module}

            sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)
            sched_module = lr_scheduler.MultiStepLR(optim_module, milestones=MILESTONES)
            schedulers = {'backbone': sched_backbone, 'module': sched_module}

            train(models, criterion, optimizers, schedulers, dataloaders, EPOCH, EPOCHL)
            acc = test(models, dataloaders, mode='test')

            fp.write(f'{acc}\n')
            print('Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {}'.format(trial + 1, TRIALS, cycle + 1,
                                                                                        CYCLES, len(labeled_set), acc))

            random.shuffle(unlabeled_set)
            subset = unlabeled_set[:SUBSET]

            unlabeled_loader = DataLoader(data_unlabeled, batch_size=BATCH,
                                          sampler=SubsetSequentialSampler(subset),
                                          pin_memory=True)
            labeled_loader = DataLoader(data_unlabeled, batch_size=BATCH,
                                        sampler=SubsetSequentialSampler(labeled_set),
                                        pin_memory=True)

            ordered_set = get_uncertainty(models, code_gen, index_idf, unlabeled_loader, labeled_loader, subset)

            labeled_set += ordered_set[:ADDENDUM]
            unlabeled_set = list(set(unlabeled_set) - set(labeled_set))

            dataloaders['train'] = DataLoader(data_train, batch_size=BATCH,
                                              sampler=SubsetRandomSampler(labeled_set),
                                              pin_memory=True)

        fp.close()