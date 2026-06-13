# -*- coding: utf-8 -*-
# General
import os
import random
import argparse
import numpy as np
import importlib

import torch
import numpy as np

# Torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler

# Torchvison
import torchvision.transforms as T
import torchvision.models as models
from torchvision.datasets import CIFAR100, CIFAR10, SVHN, ImageFolder

import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image

# Data
DATASET = 'cifar10'
DATA_DIR = 'data'
CLASS = 10
NUM_TRAIN = 50000
BATCH = 128
SUBSET = 10000
START = 500
ADDENDUM  = 500

# Active learning setting
TRIALS = 3
CYCLES = 7

# Training setting
MARGIN = 1.0
WEIGHT = 0.02
EPOCH = 200
LR = 0.1
MOMENTUM = 0.9
WDECAY = 5e-4
MILESTONES = [160]
EPOCHL = 120 # After 120 epochs, stop the gradient from the loss prediction module propagated to the target model

class SubsetSequentialSampler(torch.utils.data.Sampler):
    r"""Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape

    input = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1 # 1 operation which is defined by the authors

    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0) # Note that the size of input is already halved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()

    return loss

class Sinkhorn_sample(nn.Module):
    def __init__(self, T):
        super(Sinkhorn_sample, self).__init__()
        self.T = 2
    def sinkhorn_normalized(self,x, n_iters=10):
        for _ in range(n_iters):
            x = x / torch.sum(x, dim=1, keepdim=True)
            x = x / torch.sum(x, dim=0, keepdim=True)
        return x

    def sinkhorn_loss(self,x, y, epsilon=0.1, n_iters=20):
        Wxy = torch.cdist(x, y, p=1)
        K = torch.exp(-Wxy / epsilon)
        P = self.sinkhorn_normalized(K, n_iters)
        return torch.sum(P * Wxy)
    def forward(self, y_s, y_t, mode="classification"):
        batch_size = y_s.size(0)
        p_s = F.softmax(y_s, dim=-1)
        p_t = F.softmax(y_t, dim=-1)

        emd_loss = 0.0

        emd_loss = self.sinkhorn_loss(x=p_s,y=p_t)
        return 0.001 * emd_loss

def train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss):
    models['backbone'].train()
    if args.auxiliary == 'TOD':
        models['ema'].train()
    global iters
    Sinkhorn_loss=Sinkhorn_sample(T=2).cuda()
    for data in dataloaders['train']:
        inputs = data[0].cuda()
        labels = data[1].cuda()
        iters += 1

        optimizers['backbone'].zero_grad()

        # task loss
        scores, cons_scores, features, features_list = models['backbone'](inputs)
        target_loss = criterion(scores, labels)
        loss = torch.sum(target_loss) / target_loss.size(0)

        # unsupervised loss
        if args.auxiliary == 'TOD':
            u_inputs, _ = next(iter(dataloaders['extra']))
            u_inputs = u_inputs.cuda()
            u_scores, cons_u_scores, features_u, u_features_list = models['backbone'](u_inputs)
            ema_scores, _, _, _ = models['ema'](inputs)
            ema_u_scores, _, _, _ = models['ema'](u_inputs)
            res_loss = F.mse_loss(scores, cons_scores) + F.mse_loss(u_scores, cons_u_scores)
            consistency_loss = Sinkhorn_loss(cons_scores, ema_scores) + Sinkhorn_loss(cons_u_scores, ema_u_scores)
            loss = loss + WEIGHT * (res_loss + consistency_loss)

        loss.backward()
        optimizers['backbone'].step()
        if args.auxiliary == 'TOD':
            update_ema_variables(models['backbone'], models['ema'], 0.999, iters)

def train(models, criterion, optimizers, schedulers, dataloaders, num_epochs, epoch_loss, cycle):
    print('>> Train a Model...')
    best_acc = 0.

    for epoch in range(num_epochs):

        train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss)
        schedulers['backbone'].step()

        if epoch % 20 == 0 or epoch == 199:
            acc = test(models, dataloaders, 'test')
            if best_acc < acc:
                best_acc = acc
            print(DATASET, 'Cycle:', cycle+1, 'Epoch:', epoch, '---', 'Val Acc: {:.2f} \t Best Acc: {:.2f}'.format(acc, best_acc), flush=True)
    print('>> Finished.')

def test(models, dataloaders, mode='val'):
    assert mode == 'val' or mode == 'test'
    models['backbone'].eval()

    total = 0
    correct = 0
    with torch.no_grad():
        for (inputs, labels) in dataloaders[mode]:
            inputs = inputs.cuda()
            labels = labels.cuda()

            scores, _, _, _  = models['backbone'](inputs)
            _, preds = torch.max(scores.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return 100 * correct / total

# Function to compute Sinkhorn distance
def sinkhorn_distance(x, y, epsilon=0.01, n_iter=50):
    cost_matrix = torch.cdist(x, y, p=2) ** 2

    u = torch.zeros(x.shape[0], device=x.device)
    v = torch.zeros(y.shape[0], device=y.device)

    for _ in range(n_iter):
        u = -epsilon * torch.logsumexp((-cost_matrix + v[None, :]) / epsilon, dim=1)
        v = -epsilon * torch.logsumexp((-cost_matrix + u[:, None]) / epsilon, dim=0)

    transport_matrix = torch.exp((u[:, None] + v[None, :] - cost_matrix) / epsilon)
    return transport_matrix

from sklearn.cluster import KMeans
def cluster(embeddings):
    kmeans = KMeans(n_clusters=10)
    #kmeans.fit((embeddings.detach()).cpu())
    kmeans.fit(embeddings)
    cluster_centers = kmeans.cluster_centers_

    # Hybrid scoring (Sinkhorn + Entropy)
    selected_indices = []
    for i, center in enumerate(cluster_centers):
        dist = np.linalg.norm(embeddings - center, axis=1)
        nearest_idx = np.argmin(dist)
        selected_indices.append(nearest_idx)
    return selected_indices

class Sinkhorn(nn.Module):
    def __init__(self, T):
        super(Sinkhorn, self).__init__()
        self.T = 2
    def sinkhorn_normalized(self,x, n_iters=10):
        for _ in range(n_iters):
            x = x / torch.sum(x, dim=1, keepdim=True)
            x = x / torch.sum(x, dim=0, keepdim=True)
        return x

    def sinkhorn_transport(self,x, y, epsilon=0.1, n_iters=20):
        Wxy = torch.cdist(x, y, p=1)
        K = torch.exp(-Wxy / epsilon)
        P = self.sinkhorn_normalized(K, n_iters)
        return P
    def forward(self, y_s, y_t, mode="classification"):
        return self.sinkhorn_transport(y_s,y_t)


def compute_sinkhorn_sensitivity(models, dataloaders, unlabeled_loader, epsilon=0.01):
    """
    Compute Sinkhorn Sensitivity Score for active learning sample selection.

    Arguments:
    - models: Dictionary containing model components (expects 'backbone' key).
    - dataloaders: Dictionary containing labeled ('train') and unlabeled ('extra') data loaders.
    - unlabeled_loader: DataLoader for the full unlabeled dataset.
    - epsilon: Regularization coefficient for Sinkhorn distance.

    Returns:
    - Sensitivity scores for unlabeled samples.
    """

    # Get labeled samples
    inputs, _ = next(iter(dataloaders['train']))
    inputs = inputs.cuda()
    scores, _, f, _ = models['backbone'](inputs)

    # Get unlabeled samples from extra dataloader
    u_inputs, _ = next(iter(dataloaders['extra']))
    u_inputs = u_inputs.cuda()
    u_scores, _, f_u, _ = models['backbone'](u_inputs)

    # Compute optimal transport plan P* using Sinkhorn
    Plan=Sinkhorn(T=1).cuda()  # Should return a matrix
    P_star= Plan(scores,u_scores)
    #print(len(unlabeled_loader.dataset))
    # Initialize sensitivity scores
    num_unlabeled = len(unlabeled_loader)
    sensitivity_scores = np.zeros(num_unlabeled)
    uncertainty = torch.tensor([]).cuda()

    # Compute sensitivity for each unlabeled sample
    for batch_idx, (u_star, _) in enumerate(unlabeled_loader):  # Batch index
        u_star = u_star.cuda()
        u_star_scores, _, _, _ = models['backbone'](u_star)# Get features
        u_star_s=u_star_scores
        #s_idx= cluster((u_star_s.detach()).cpu())
        #print(len(s_idx))
        for sample_idx in range(len(u_star)):  # Loop through batch samples
            grad_W = torch.zeros_like(scores[0]).cuda()  # Gradient accumulator

            for i in range(len(inputs)):  # Loop through labeled samples
                if sample_idx < P_star.shape[1]:  # Ensure valid indexing
                    grad_W += -2 * P_star[i, sample_idx] * (scores[i] - u_star_scores[sample_idx])

            uncertainty = torch.cat((uncertainty, (torch.norm(grad_W).detach()).reshape(1)), dim=0)
            #print(len(uncertainty))
    #uncertainty = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min())
    return uncertainty.cpu()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Semi-Supervised Active Learning')
    #parser.add_argument('--config', default='cifar10', type=str, help='dataset config path')
    parser.add_argument('--sampling', default='TOD', type=str, help='data sampling method', choices=['RANDOM', 'TOD'])
    parser.add_argument('--auxiliary', default='TOD', type=str, help='auxiliary training loss', choices=['NONE', 'TOD'])
    #args = parser.parse_args()
    #args, unknown = parser.parse_known_args()
    args = parser.parse_args(args=[])
    #config = importlib.import_module('config.'+args.config)
    #config.SAMPLING = args.sampling # Random | TOD
    #config.AUXILIARY = args.auxiliary # NONE | TOD
    #to_import = [name for name in dir(config) if not name.startswith('_')]
    #globals().update({name: getattr(config, name) for name in to_import})

    # Data
    if DATASET == 'cifar10':
        train_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(size=32, padding=4),
            T.ToTensor(),
            T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])
        test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])

        data_train = CIFAR10(DATA_DIR, train=True, download=True, transform=train_transform)
        data_unlabeled = CIFAR10(DATA_DIR, train=True, download=True, transform=test_transform)
        data_test = CIFAR10(DATA_DIR, train=False, download=True, transform=test_transform)

    elif DATASET == 'cifar100':
        train_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(size=32, padding=4),
            T.ToTensor(),
            T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])
        test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])
        data_train = CIFAR100(DATA_DIR, train=True, download=True, transform=train_transform)
        data_unlabeled = CIFAR100(DATA_DIR, train=True, download=True, transform=test_transform)
        data_test = CIFAR100(DATA_DIR, train=False, download=True, transform=test_transform)

    elif DATASET == 'svhn':
        train_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(size=32, padding=4),
            T.ToTensor(),
            T.Normalize([0.4310, 0.4302, 0.4463], [0.1965, 0.1984, 0.1992])
        ])
        test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.4310, 0.4302, 0.4463], [0.1965, 0.1984, 0.1992])
        ])

        data_train = SVHN(root=DATA_DIR, split='train', transform=train_transform, download=True)
        data_unlabeled = SVHN(root=DATA_DIR, split='train', transform=train_transform, download=True)
        data_test = SVHN(root=DATA_DIR, split='test', transform=test_transform, download=True)



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.linear1 = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = F.avg_pool2d(out4, 4)
        out5 = out5.view(out5.size(0), -1)     # [128, 512]
        out = self.linear(out5)
        out_cons = self.linear1(out5)
        return out, out_cons, out5, [out1, out2, out3, out4]


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2,2,2,2], num_classes)

for trial in range(TRIALS):
        global iters
        iters = 0

        indices = list(range(NUM_TRAIN))
        random.shuffle(indices)
        labeled_set = indices[:START]
        unlabeled_set = indices[START:]

        train_loader = DataLoader(data_train, batch_size=BATCH,     # BATCH
                                  sampler=SubsetRandomSampler(labeled_set),
                                  pin_memory=True)
        test_loader = DataLoader(data_test, batch_size=BATCH)
        extra_loader = DataLoader(data_train, batch_size=BATCH,
                                  sampler=SubsetSequentialSampler(unlabeled_set),
                                  pin_memory=True)

        dataloaders = {'train': train_loader, 'test': test_loader, 'extra': extra_loader}

        backbone_net = ResNet18(num_classes=CLASS).cuda()
        cod_model = ResNet18(num_classes=CLASS).cuda()
        ema_model = ResNet18(num_classes=CLASS).cuda()

        for param in cod_model.parameters():
            param.detach_()
        for param in ema_model.parameters():
            param.detach_()

        models = {'backbone': backbone_net, 'ema': ema_model, 'cod': cod_model}
        torch.backends.cudnn.benchmark = True

         # Active learning cycles
        for cycle in range(CYCLES):

            if cycle > 0:
                checkpoint = torch.load('./weights/{}_auxiliary_{}_sampling_{}_trial{}_cycle{}.pth'.format(DATASET,args.auxiliary ,args.sampling, trial+1, cycle))
                models['cod'].load_state_dict(checkpoint['state_dict_backbone'])

            # Loss, criterion and scheduler (re)initialization
            criterion = nn.CrossEntropyLoss(reduction='none')

            optim_backbone = optim.SGD(models['backbone'].parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WDECAY)
            sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)

            optimizers = {'backbone': optim_backbone}
            schedulers = {'backbone': sched_backbone}

            # Training and test
            train(models, criterion, optimizers, schedulers, dataloaders, EPOCH, EPOCHL, cycle)
            acc = test(models, dataloaders, mode='test')
            print('{} auxiliary:{} sampling:{} Trial:{}/{} || Cycle:{}/{} || Label set size:{} ||  Test acc:{:.2f}'.format(DATASET,args.auxiliary,args.sampling, trial+1, TRIALS, cycle+1, CYCLES, len(labeled_set), acc), flush=True)

            # Active sampling
            random.shuffle(unlabeled_set)

            if args.sampling == 'RANDOM':
                subset = unlabeled_set[:ADDENDUM]
                labeled_set += subset
                unlabeled_set = unlabeled_set[ADDENDUM:]
            else:
                subset = unlabeled_set[:SUBSET]

                # Create unlabeled dataloader for the unlabeled subset
                unlabeled_loader = DataLoader(data_unlabeled, batch_size=BATCH,
                                              sampler=SubsetSequentialSampler(subset),
                                              pin_memory=True)
                print(len(unlabeled_loader))
                # Measure uncertainty of each data points in the subset
                uncertainty = compute_sinkhorn_sensitivity(models, dataloaders, unlabeled_loader, epsilon=0.01)
                #print(len(uncertainty))
                # Index in ascending order
                arg = np.argsort(uncertainty)
                #print(max(arg))
                # Update the labeled dataset and the unlabeled dataset, respectively
                if cycle > 0:
                    labeled_set += list(torch.tensor(subset)[arg][-ADDENDUM:].numpy())
                    unlabeled_set = list(torch.tensor(subset)[arg][:-ADDENDUM].numpy()) + unlabeled_set[SUBSET:]
                else:
                    labeled_set += list(torch.tensor(subset)[arg][:ADDENDUM].numpy())
                    unlabeled_set = list(torch.tensor(subset)[arg][ADDENDUM:].numpy()) + unlabeled_set[SUBSET:]

            # Create a new dataloader for the updated labeled dataset
            dataloaders['train'] = DataLoader(data_train, batch_size=BATCH,
                                              sampler=SubsetRandomSampler(labeled_set),
                                              pin_memory=True)
            dataloaders['extra'] = DataLoader(data_train, batch_size=BATCH,
                                              sampler=SubsetRandomSampler(unlabeled_set),
                                              pin_memory=True)

            if not os.path.exists('weights'):
                os.makedirs('weights')
            torch.save({
                    'cycle': cycle + 1,
                    'state_dict_backbone': models['backbone'].state_dict()
                },
                './weights/{}_auxiliary_{}_sampling_{}_trial{}_cycle{}.pth'.format(DATASET,args.auxiliary,args.sampling, trial+1, cycle+1))
            print('finished')
