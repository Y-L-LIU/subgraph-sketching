"""
training functions
"""
import time
from math import inf

import torch
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm
import wandb
import numpy as np
import random
from src.utils import get_num_samples


def get_train_func(args):
    if args.model == 'ELPH':
        return train_elph
    elif args.model == 'BUDDY':
        train_func = train_buddy
    else:
        train_func = train
    return train_func


def train_buddy(model, optimizer, train_loader, args, device, emb=None):
    print('starting training')
    t0 = time.time()
    model.train()
    total_loss = 0
    data = train_loader.dataset
    # hydrate edges
    links = data.links
    labels = torch.tensor(data.labels)
    marks = torch.tensor(data.marks)
    # sampling
    train_samples = get_num_samples(args.train_samples, len(labels))
    sample_indices = torch.randperm(len(labels))[:train_samples]
    links = links[sample_indices]
    labels = labels[sample_indices]
    marks = marks[sample_indices]
    if args.wandb:
        wandb.log({"train_total_batches": len(train_loader)})
    batch_processing_times = []
    loader = DataLoader(range(len(links)), args.batch_size, shuffle=True)
    for batch_count, indices in enumerate(tqdm(loader)):
        # do node level things
        if model.node_embedding is not None:
            if args.propagate_embeddings:
                emb = model.propagate_embeddings_func(data.edge_index.to(device))
            else:
                emb = model.node_embedding.weight
        else:
            emb = None
        curr_links = links[indices]
        batch_emb = None if emb is None else emb[curr_links].to(device)

        if args.use_struct_feature:
            sf_indices = sample_indices[indices]  # need the original link indices as these correspond to sf
            subgraph_features = data.subgraph_features[sf_indices].to(device)
        else:
            subgraph_features = torch.zeros(data.subgraph_features[indices].shape).to(device)
        node_features = data.x[curr_links].to(device)
        degrees = data.degrees[curr_links].to(device)
        if args.use_RA:
            ra_indices = sample_indices[indices]
            RA = data.RA[ra_indices].to(device)
        else:
            RA = None
        start_time = time.time()
        optimizer.zero_grad()
        curr_marks = marks[indices]
        logits = model(subgraph_features, node_features, degrees[:, 0], degrees[:, 1], RA, batch_emb)
        loss = get_loss(args.loss)(logits, labels[indices].squeeze(0).to(device))

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * args.batch_size
        batch_processing_times.append(time.time() - start_time)

    if args.wandb:
        wandb.log({"train_batch_time": np.mean(batch_processing_times)})
        wandb.log({"train_epoch_time": time.time() - t0})

    print(f'training ran in {time.time() - t0}')

    if args.log_features:
        model.log_wandb()

    return total_loss / len(train_loader.dataset)


def train(model, optimizer, train_loader, args, device, emb=None):
    """
    Adapted version of the SEAL training function
    :param model:
    :param optimizer:
    :param train_loader:
    :param args:
    :param device:
    :param emb:
    :return:
    """

    print('starting training')
    t0 = time.time()
    model.train()
    if args.dynamic_train:
        train_samples = get_num_samples(args.train_samples, len(train_loader.dataset))
    else:
        train_samples = inf
    total_loss = 0
    pbar = tqdm(train_loader, ncols=70)
    if args.wandb:
        wandb.log({"train_total_batches": len(train_loader)})
    batch_processing_times = []
    data = train_loader.dataset
    # print(data.x.shape)
    links = data.links
    marks = data.marks
    mark1 = data.all_mark1.to(device)
    mark1_label = data.all_mark1_label.to(device)
    labels = torch.tensor(data.labels)
    # sampling
    if args.dynamic_train:
        train_samples = get_num_samples(args.train_samples, len(labels))
        sample_indices = torch.randperm(len(labels))[:train_samples]
        links = links[sample_indices]
        labels = labels[sample_indices]
        marks = data.marks[sample_indices]

    loader = DataLoader(range(len(links)), args.batch_size, shuffle=True)
    for batch_count, indices in enumerate(tqdm(loader)):
        start_time = time.time()
        optimizer.zero_grad()
        # todo this loop should no longer be hit as this function isn't called for BUDDY
        if args.model == 'BUDDY':
            data_dev = [elem.squeeze().to(device) for elem in data]
            logits = model(*data_dev[:-1])  # everything but the labels
            loss = get_loss(args.loss)(logits, data[-1].squeeze(0).to(device))
        elif 'SEAL' in args.model:
            data = data.to(device)
            x = data.x if args.use_feature else None
            edge_weight = data.edge_weight if args.use_edge_weight else None
            node_id = data.node_id if emb else None
            logits = model(data.z, data.edge_index, data.batch, x, edge_weight, node_id, data.src_degree,
                           data.dst_degree)
            loss = get_loss(args.loss)(logits, data.y)
        else:
            node_features = model(emb.weight.to(device), data.edge_index.to(device))
            curr_links = links[indices].to(device)
            curr_labels = labels[indices].to(device)
            curr_marks = marks[indices].to(device)
            num_sample=0
            if args.dblp and num_sample:
                upsample_index = torch.randperm(mark1.size(0))[:num_sample]
                upsample_links = mark1[upsample_index]
                upsample_labels = mark1_label[upsample_index]
                curr_links = torch.concat([curr_links,upsample_links], dim=0)
                curr_labels = torch.concat([curr_labels, upsample_labels], dim=0)
                curr_marks = torch.concat([curr_marks,torch.ones(num_sample).to(device)])
            num_ipm = 128
            batch_node_features = None if node_features is None else node_features[curr_links]
            optimizer.zero_grad()
            if args.dblp:
                logits = model.predictor(batch_node_features, curr_marks)
                pos_weight = (curr_labels.size(0)-curr_labels.sum())/curr_labels.sum()
                loss = get_loss(args.loss)(logits, curr_labels.squeeze(0).to(device),pos_weight=pos_weight)
                selected_t0_link = curr_links[curr_marks==0][:num_ipm]
                current_t0_link = node_features[selected_t0_link]
                current_t0_link_emb = current_t0_link[:, 0, :] * current_t0_link[:, 1, :]
                selected_t1_links = mark1[torch.randperm(mark1.size(0))][:num_ipm]
                t1_link = node_features[selected_t1_links]
                current_t1_link_emb = t1_link[:, 0, :] * t1_link[:, 1, :]
                # Calculate MMD
                disc = mmd(current_t0_link_emb, current_t1_link_emb)    
                loss = loss + 0.05 * disc   
            else:
                pos_weight = (curr_labels.size(0)-curr_labels.sum())/curr_labels.sum()
                logits = model.predictor(batch_node_features)
                loss = get_loss(args.loss)(logits, curr_labels.squeeze(0).to(device),pos_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(emb.weight, 1.0)

        optimizer.step()
        total_loss += loss.item() * args.batch_size
        torch.cuda.empty_cache()
        batch_processing_times.append(time.time() - start_time)
    if args.wandb:
        wandb.log({"train_batch_time": np.mean(batch_processing_times)})
        wandb.log({"train_epoch_time": time.time() - t0})

    print(f'training ran in {time.time() - t0}')
    if args.model in {'linear', 'pmi', 'ra', 'aa', 'one_layer'}:
        model.print_params()

    if args.log_features:
        model.log_wandb()

    return total_loss / len(train_loader.dataset)


def mmd(x, y, kernel_type='rbf', kernel_param=1.0):
    # Compute pairwise squared Euclidean distances
    xx = torch.cdist(x, x, p=2).pow(2)
    yy = torch.cdist(y, y, p=2).pow(2)
    xy = torch.cdist(x, y, p=2).pow(2)

    # Compute kernel function
    if kernel_type == 'rbf':
        xx_kernel = torch.exp(-xx / (2 * kernel_param ** 2))
        yy_kernel = torch.exp(-yy / (2 * kernel_param ** 2))
        xy_kernel = torch.exp(-xy / (2 * kernel_param ** 2))
    elif kernel_type == 'linear':
        xx_kernel = xx
        yy_kernel = yy 
        xy_kernel = xy
    else:
        raise ValueError("Invalid kernel type. Supported types: 'rbf', 'linear'.")
    mmd = xx_kernel.mean() + yy_kernel.mean() - 2 * xy_kernel.mean()
    return mmd

def train_elph(model, optimizer, train_loader, args, device,emb1=None):
    """
    train a GNN that calculates hashes using message passing
    @param model:
    @param optimizer:
    @param train_loader:
    @param args:
    @param device:
    @return:
    """
    print('starting training')
    t0 = time.time()
    model.train()
    total_loss = 0
    data = train_loader.dataset
    # hydrate edges
    links = data.links
    mark1 = data.all_mark1.to(device)
    mark1_label = data.all_mark1_label.to(device)
    labels = torch.tensor(data.labels)
    # sampling
    train_samples = get_num_samples(args.train_samples, len(labels))
    sample_indices = torch.randperm(len(labels))[:train_samples]
    links = links[sample_indices]
    labels = labels[sample_indices]
    marks = data.marks[sample_indices]
    if args.wandb:
        wandb.log({"train_total_batches": len(train_loader)})
    batch_processing_times = []

    loader = DataLoader(range(len(links)), args.batch_size, shuffle=True)
    for batch_count, indices in enumerate(tqdm(loader)):
        # do node level things
        if model.node_embedding is not None:
            if args.propagate_embeddings:
                emb = model.propagate_embeddings_func(data.edge_index.to(device))
            else:
                emb = model.node_embedding.weight
        else:
            emb = None
        # get node features
        node_features, hashes, cards = model(data.x.to(device), data.edge_index.to(device))
        # print(node_features)
        curr_links = links[indices].to(device)
        curr_labels = labels[indices].to(device)
        curr_marks = marks[indices].to(device)
        num_sample=0
        if args.dblp and num_sample:
            upsample_index = torch.randperm(mark1.size(0))[:num_sample]
            upsample_links = mark1[upsample_index]
            upsample_labels = mark1_label[upsample_index]
            curr_links = torch.concat([curr_links,upsample_links], dim=0)
            curr_labels = torch.concat([curr_labels, upsample_labels], dim=0)
            curr_marks = torch.concat([curr_marks,torch.ones(num_sample).to(device)])
        num_ipm = 128
        batch_node_features = None if node_features is None else node_features[curr_links]
        batch_emb = None if emb is None else emb[curr_links].to(device)
        # hydrate link features
        if args.use_struct_feature:
            subgraph_features = model.elph_hashes.get_subgraph_features(curr_links, hashes, cards).to(device)
        else:  # todo fix this
            subgraph_features = torch.zeros(data.subgraph_features[indices].shape).to(device)
        start_time = time.time()
        optimizer.zero_grad()
        table = emb if emb is not None else node_features
        if args.dblp:
            logits = model.predictor(subgraph_features,batch_node_features, batch_emb,curr_marks)
            pos_weight = (curr_labels.size(0)-curr_labels.sum())/curr_labels.sum()
            loss = get_loss(args.loss)(logits, curr_labels.squeeze(0).to(device),pos_weight=pos_weight)
            selected_t0_link = curr_links[curr_marks==0][:num_ipm]
            current_t0_link = table[selected_t0_link]
            current_t0_link_emb = current_t0_link[:, 0, :] * current_t0_link[:, 1, :]
            selected_t1_links = mark1[torch.randperm(mark1.size(0))][:num_ipm]
            t1_link = table[selected_t1_links]
            current_t1_link_emb = t1_link[:, 0, :] * t1_link[:, 1, :]
            # Calculate MMD
            disc = mmd(current_t0_link_emb, current_t1_link_emb)    
            loss = loss + 0.01 * disc
        else:
            logits = model.predictor(subgraph_features, batch_node_features, batch_emb)
            loss = get_loss(args.loss)(logits, curr_labels.squeeze(0).to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * args.batch_size
        batch_processing_times.append(time.time() - start_time)

    if args.wandb:
        wandb.log({"train_batch_time": np.mean(batch_processing_times)})
        wandb.log({"train_epoch_time": time.time() - t0})

    print(f'training ran in {time.time() - t0}')
    if args.model in {'linear', 'pmi', 'ra', 'aa', 'one_layer'}:
        model.print_params()

    if args.log_features:
        model.log_wandb()

    return total_loss / len(train_loader.dataset)


def auc_loss(logits, y, num_neg=1):
    pos_out = logits[y == 1]
    neg_out = logits[y == 0]
    # hack, should really pair negative and positives in the training set
    if len(neg_out) <= len(pos_out):
        pos_out = pos_out[:len(neg_out)]
    else:
        neg_out = neg_out[:len(pos_out)]
    pos_out = torch.reshape(pos_out, (-1, 1))
    neg_out = torch.reshape(neg_out, (-1, num_neg))
    return torch.square(1 - (pos_out - neg_out)).sum()


def bce_loss(logits, y, pos_weight=None):
    return BCEWithLogitsLoss(pos_weight=pos_weight)(logits.view(-1), y.to(torch.float))


def get_loss(loss_str):
    if loss_str == 'bce':
        loss = bce_loss
    elif loss_str == 'auc':
        loss = auc_loss
    else:
        raise NotImplementedError
    return loss
