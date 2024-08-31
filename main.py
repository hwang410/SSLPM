import os
import pickle
import random
import argparse
from sklearn.cluster import AgglomerativeClustering

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler

from tqdm import tqdm

from config import *
from train_utils import frozen, free
from models.resnet import ResNet18
from models.lossnet import LossNet
from data.data_transform import get_data
from data.sampler import SubsetSequentialSampler


def loss_pred_loss(input, target, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    criterion = nn.BCELoss()
    input = (input - input.flip(0))[:len(input) // 2]
    target = (target - target.flip(0))[:len(target) // 2]
    target = target.detach()
    diff = torch.sigmoid(input)
    one = torch.sign(torch.clamp(target, min=0))

    if reduction == 'mean':
        loss = criterion(diff, one)
    elif reduction == 'none':
        loss = criterion(diff, one)
    else:
        NotImplementedError()

    return loss


def ft_epoch_wich_ss(models, criterions, optimizers, dataloaders):
    models['backbone'].eval()
    models['module'].train()

    free(models['module'])
    frozen(models['backbone'])

    ul_iter = iter(dataloaders['semi'])
    for data in tqdm(dataloaders['ft'], leave=False, total=len(dataloaders['ft'])):
        inputs_l = data[0].cuda()
        labels_l = data[1].cuda()

        optimizers['ft'].zero_grad()

        scores_l, features_l, _ = models['backbone'](inputs_l)

        features_l[0] = features_l[0].detach()
        features_l[1] = features_l[1].detach()
        features_l[2] = features_l[2].detach()
        features_l[3] = features_l[3].detach()

        pred_loss = models['module'](features_l)
        pred_loss = pred_loss.view(pred_loss.size(0))

        target_loss = criterions['ce'](scores_l, labels_l)
        module_loss = loss_pred_loss(pred_loss, target_loss)  # predictor loss - labeled data

        ##################################################################################

        inputs_ul, _ = next(ul_iter)
        inputs_ul_w, inputs_ul_s = inputs_ul
        inputs_ul_w = inputs_ul_w.cuda()
        inputs_ul_s = inputs_ul_s.cuda()

        scores_ul_w, features_ul_w, _ = models['backbone'](inputs_ul_w)
        scores_ul_s, features_ul_s, _ = models['backbone'](inputs_ul_s)

        features_ul_s[0] = features_ul_s[0].detach()
        features_ul_s[1] = features_ul_s[1].detach()
        features_ul_s[2] = features_ul_s[2].detach()
        features_ul_s[3] = features_ul_s[3].detach()

        # Pseudo labeling (based on posterior probability of weakly augmented data)
        pseudo_label = torch.softmax(scores_ul_w.detach() / 1, dim=-1)
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(THRESHOLD).float()
        indices = torch.nonzero(mask).reshape(-1,)
        indices = indices[:-1] if indices.size(0) % 2 else indices

        # Loss prediction for strongly augmented data
        pred_loss_s = models['module'](features_ul_s)
        pred_loss_s = pred_loss_s.view(pred_loss_s.size(0))
        strong_pred_loss_threshold = torch.index_select(pred_loss_s, dim=0, index=indices)

        # pseudo loss
        strong_scores_threshold = torch.index_select(scores_ul_s, dim=0, index=indices)
        strong_target_threshold = torch.index_select(pseudo_label, dim=0, index=indices)
        pseudo_loss = criterions['ce'](strong_scores_threshold, strong_target_threshold)

        pseudo_module_loss = loss_pred_loss(strong_pred_loss_threshold, pseudo_loss)  # predictor loss - unlabeled data

        ##################################################################################

        loss = module_loss + pseudo_module_loss

        loss.backward()
        optimizers['ft'].step()


def train_epoch(models, criterions, optimizers, dataloaders, epoch, epoch_loss):
    models['backbone'].train()
    models['module'].train()

    free(models['module'])
    free(models['backbone'])

    for data in tqdm(dataloaders['labeled'], leave=False, total=len(dataloaders['labeled'])):
        inputs_l = data[0].cuda()
        labels_l = data[1].cuda()

        optimizers['backbone'].zero_grad()
        optimizers['module'].zero_grad()

        scores_l, features_l, _ = models['backbone'](inputs_l)

        if epoch > epoch_loss:
            features_l[0] = features_l[0].detach()
            features_l[1] = features_l[1].detach()
            features_l[2] = features_l[2].detach()
            features_l[3] = features_l[3].detach()
        pred_loss = models['module'](features_l)
        pred_loss = pred_loss.view(pred_loss.size(0))

        target_loss = criterions['ce'](scores_l, labels_l)

        backbone_loss = torch.sum(target_loss) / target_loss.size(0)  # labeled data loss

        module_loss = loss_pred_loss(pred_loss, target_loss)  # predictor loss

        loss = backbone_loss + (module_loss * WEIGHT)

        loss.backward()
        optimizers['backbone'].step()
        optimizers['module'].step()


def train(models, criterions, optimizers, schedulers, dataloaders, num_epochs, epoch_loss, cycle):
    print('>> Train a Model.')

    for epoch in range(num_epochs):
        train_epoch(models, criterions, optimizers, dataloaders, epoch, epoch_loss)

        schedulers['backbone'].step()
        schedulers['module'].step()

    if cycle < CYCLES - 1:
        for epoch in range(30):
            ft_epoch_wich_ss(models, criterions, optimizers, dataloaders)
            schedulers['ft'].step()

    print('>> Finished.')
    

def test(models, dataloaders):
    models['backbone'].eval()
    models['module'].eval()

    total = 0
    correct = 0
    with torch.no_grad():
        for (inputs, labels) in dataloaders['test']:
            inputs = inputs.cuda()
            labels = labels.cuda()

            scores, _, _ = models['backbone'](inputs)
            _, preds = torch.max(scores.data, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return 100 * correct / total


def get_uncertainty(models, unlabeled_loader):
    models['backbone'].eval()
    models['module'].eval()

    uncertainty = torch.tensor([]).cuda()
    with torch.no_grad():
        for (inputs, labels) in unlabeled_loader:
            inputs = inputs.cuda()

            scores, features, _ = models['backbone'](inputs)
            pred_loss = models['module'](features)
            pred_loss = pred_loss.view(pred_loss.size(0))

            uncertainty = torch.cat((uncertainty, pred_loss), 0)

    return uncertainty.cpu()


def get_real_loss(models, data_loader, criterions):
    models['backbone'].eval()
    models['module'].eval()

    uncertainty = torch.tensor([]).cuda()
    with torch.no_grad():
        for (inputs, labels) in data_loader:
            inputs = inputs.cuda()
            labels = labels.cuda()

            scores, features, _ = models['backbone'](inputs)
            target_loss = criterions['ce'](scores, labels)

            uncertainty = torch.cat((uncertainty, target_loss), 0)

    return uncertainty.cpu()


def clustering(model, cluster_size, data_loader):
    model.eval()

    features = torch.tensor([]).cuda()
    with torch.no_grad():
        for (inputs, labels) in data_loader:
            inputs = inputs.cuda()

            _, _, feature = model(inputs)

            features = torch.cat((features, feature), 0)
    features = features.cpu().numpy()

    return AgglomerativeClustering(n_clusters=cluster_size, linkage='complete').fit_predict(features)


def sampling(cluster_dict):
    sampled = []
    for key in cluster_dict:
        sampled.append(cluster_dict[key][-1])

    return sampled


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    sd = args.seed

    random.seed(9410+sd)
    np.random.seed(9410+sd)
    torch.manual_seed(9410+sd)
    torch.cuda.manual_seed(9410+sd)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    labeled_set = [i for i in range(50000)]
    random.shuffle(labeled_set)
    unlabeled_set = labeled_set[2500:]
    labeled_set = labeled_set[:2500]

    loss_module = LossNet().cuda()
    backbone = ResNet18(num_classes=CLS_CNT, channel_size=CHANNEL_SIZE).cuda()
    models = {'backbone': backbone, 'module': loss_module}

    train_transform_data, ssl_transform_data, test_transform_data, evaluate_transform_data = get_data('./data', DATASET)

    semi_batch_size = len(unlabeled_set) // (len(labeled_set) // BATCH + 1)
    semi_loader = DataLoader(ssl_transform_data,
                             batch_size=min(semi_batch_size - 1 if semi_batch_size % 2 else semi_batch_size, 1536),
                             sampler=SubsetRandomSampler(unlabeled_set),
                             pin_memory=True)
    test_loader = DataLoader(test_transform_data, batch_size=BATCH)
    ft_loader = DataLoader(train_transform_data, batch_size=BATCH,sampler=SubsetRandomSampler(labeled_set),
                           pin_memory=True)

    removal_size = (len(labeled_set) // 100) - 1 if (len(labeled_set) // 100) % 2 else (len(labeled_set) // 100)
    labeled_loader = DataLoader(train_transform_data, batch_size=BATCH,
                                sampler=SubsetRandomSampler(labeled_set[:-removal_size]),
                                pin_memory=True)
    dataloaders = {'labeled': labeled_loader, 'test': test_loader, 'semi': semi_loader, 'ft': ft_loader}

    acc_per_class = []
    for cycle in range(CYCLES):
        mse_loss = nn.MSELoss(reduction='none').cuda()
        ce_loss = nn.CrossEntropyLoss(reduction='none').cuda()
        criterions = {'ce': ce_loss, 'mse': mse_loss}

        optim_backbone = optim.SGD(models['backbone'].parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WDECAY)
        optim_module = optim.SGD(models['module'].parameters(), lr=0.05, momentum=MOMENTUM, weight_decay=WDECAY)
        optim_ft = optim.SGD(models['module'].parameters(), lr=0.01, momentum=MOMENTUM, weight_decay=WDECAY)
        optimizers = {'backbone': optim_backbone, 'module': optim_module, 'ft': optim_ft}

        sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)
        sched_module = lr_scheduler.MultiStepLR(optim_module, milestones=MILESTONES)
        sched_ft = lr_scheduler.MultiStepLR(optim_ft, milestones=[50])
        schedulers = {'backbone': sched_backbone, 'module': sched_module, 'ft': sched_ft}

        print(f'labeled: {len(labeled_set)} / unlabeled: {len(unlabeled_set)}')
        train(models, criterions, optimizers, schedulers, dataloaders, EPOCH, EPOCHL, cycle)
        acc = test(models, dataloaders)
        print(f'Cycle {cycle + 1}/{CYCLES} || Label set size {len(labeled_set)}: Test acc {acc}')

        if cycle < CYCLES - 1:
            unlabeled_loader = DataLoader(evaluate_transform_data, batch_size=BATCH,
                                          sampler=SubsetSequentialSampler(unlabeled_set),
                                          pin_memory=True)

            uncertainty = get_uncertainty(models, unlabeled_loader)
            arg = np.argsort(uncertainty)
            subset_ = list(torch.tensor(unlabeled_set)[arg][-SUBSET:].numpy())
            subset_label = clustering(models['backbone'], ADDENDUM,
                                      DataLoader(evaluate_transform_data, batch_size=BATCH,
                                                 sampler=SubsetSequentialSampler(subset_),
                                                 pin_memory=True))
            subset_cluster = {}
            for i, idx in enumerate(subset_):
                if subset_label[i] not in subset_cluster:
                    subset_cluster[subset_label[i]] = [idx]
                else:
                    subset_cluster[subset_label[i]].append(idx)

            sampled_data = sampling(subset_cluster)
            sampled_loader = DataLoader(evaluate_transform_data, batch_size=BATCH,
                                        sampler=SubsetSequentialSampler(sampled_data),
                                        pin_memory=True)
            sampled_real_loss = get_real_loss(models, sampled_loader, criterions)
            sampled_arg = np.argsort(sampled_real_loss)
            sampled_data = list(torch.tensor(sampled_data)[sampled_arg].numpy())[::-1]

            labeled_loader = DataLoader(evaluate_transform_data, batch_size=BATCH,
                                        sampler=SubsetSequentialSampler(labeled_set),
                                        pin_memory=True)
            labeled_real_loss = get_real_loss(models, labeled_loader, criterions)
            labeled_arg = np.argsort(labeled_real_loss)
            labeled_set = list(torch.tensor(labeled_set)[labeled_arg].numpy())

            labeled_set += sampled_data
            unlabeled_set = list(set(unlabeled_set) - set(labeled_set))


        dataloaders['labeled'] = DataLoader(train_transform_data, batch_size=BATCH,
                                            sampler=SubsetRandomSampler(labeled_set), pin_memory=True)
        if cycle < CYCLES - 2:
            _size = removal_size // 2
            dataloaders['labeled'] = DataLoader(train_transform_data, batch_size=BATCH,
                                                sampler=SubsetRandomSampler(labeled_set[_size:-_size]),
                                                pin_memory=True)
            dataloaders['ft'] = DataLoader(train_transform_data, batch_size=BATCH,
                                           sampler=SubsetRandomSampler(labeled_set), pin_memory=True)

        semi_batch_size = len(unlabeled_set) // (len(labeled_set) // BATCH + 1)
        dataloaders['semi'] = DataLoader(ssl_transform_data,
                                         batch_size=min(semi_batch_size - 1 if semi_batch_size % 2 else semi_batch_size, 1536),
                                         sampler=SubsetRandomSampler(unlabeled_set), pin_memory=True)