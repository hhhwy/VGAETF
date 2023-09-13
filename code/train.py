import argparse
import logging
import os
import time
import random
import dgl
import pandas as pd
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import torch
import torch.nn.functional as F
import torch.nn as nn
from util import *
import model
from sklearn.model_selection import KFold
from sklearn import metrics
parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--epochs', '-e', type=int, default=1200, help='Number of epochs to train.')
parser.add_argument('--hidden1', '-h1', type=int, default=384, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', '-h2', type=int, default=320, help='Number of units in hidden layer 2.')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU id to use.')
parser.add_argument("--alpha", type=int, default=0.6)
parser.add_argument("--save_every", type=int, default=20)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument('--patience', type=int, default=30, help="used for early stop")
parser.add_argument("--evaluate_every", type=int, default=10)
args = parser.parse_args()

# check device
device = torch.device("cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu")

use_cuda = args.gpu >= 0 and torch.cuda.is_available()


def get_eval_auc_pr_metric(labels, triplets_score):
    y_true = labels.detach().cpu().numpy()
    y_true = y_true.astype(np.float32)
    pred = triplets_score.detach().cpu().numpy()
    pred = pred.astype(np.float32)
    # metrics
    auc = metrics.roc_auc_score(y_true, pred)
    precision, recall, _ = metrics.precision_recall_curve(y_true, pred)
    auprc = metrics.auc(recall, precision)

    return [auc, auprc]


def set_seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    seed = 1234
    set_seed_all(seed)
    features, train_val_triplets, test_pos_triplets, drug_node_num, relation_num = load_data()
    # -----split triplets into 5CV,test set
    test_triplets, test_labels = get_test_tri_label(test_pos_triplets, drug_node_num, relation_num)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    fold = 0
    final_metric = np.zeros(2)
    for train_edges_idx, val_edges_idx in kf.split(train_val_triplets):
        fold = fold + 1
        valid_pos_triplets = train_val_triplets[val_edges_idx]
        train_pos_triplets = train_val_triplets[train_edges_idx]
        train_pos_triplets = get_symmetry_triplets(train_pos_triplets)
        valid_pos_triplets = get_symmetry_triplets(valid_pos_triplets)
        train_graph = generate_train_graph(train_pos_triplets)
        train_false_triplets = generate_negative_data(train_pos_triplets, drug_node_num, relation_num)
        valid_false_triplets = generate_negative_data(valid_pos_triplets, drug_node_num, relation_num)
        train_triplets = get_triplets(train_pos_triplets, train_false_triplets)
        val_triplets = get_triplets(valid_pos_triplets, valid_false_triplets)
        train_labels = get_labels(train_pos_triplets, train_false_triplets)
        val_labels = get_labels(valid_pos_triplets, valid_false_triplets)
        #
        adj_d_j, adj_j_j = generate_all_heteroGraph(train_graph, drug_node_num, relation_num)
        train_graph = dgl.from_networkx(train_graph)
        train_graph = dgl.to_bidirected(train_graph)
        train_graph = train_graph.to(device)
        d_t_arr = adj_d_j.numpy()
        d_t_edges = np.where(d_t_arr == 1)
        d_t_pos = torch.tensor(d_t_arr[d_t_edges])
        d_t_pos = d_t_pos.to(device)
        t_t_arr = adj_j_j.numpy()
        t_t_edges = np.where(t_t_arr == 1)
        t_t_pos = torch.tensor(t_t_arr[t_t_edges])
        t_t_pos = t_t_pos.to(device)
        #
        features = features.to(device)
        # create model
        vgae_model = model.VGAEModel(drug_node_num, drug_node_num, relation_num, args.hidden1, args.hidden2)
        vgae_model = vgae_model.to(device)
        # create training component
        optimizer = torch.optim.Adam(vgae_model.parameters(), lr=args.learning_rate)
        print('Total Parameters:', sum([p.nelement() for p in vgae_model.parameters()]))
        model_state_file = 'model_state.pth'
        counter = 0
        best_metric = [0, 0]
        for epoch in range(args.epochs):
            vgae_model.train()
            train_triplets = train_triplets.to(device)
            triplets_score, d_t_recd, t_t_recd, reg_loss, entity_embedding = vgae_model(train_graph, features,
                                                                                        drug_node_num, train_triplets)
            loss_triplets = F.binary_cross_entropy_with_logits(triplets_score, train_labels) + reg_loss
            d_t_parr = d_t_recd.cpu().detach().numpy()
            d_t_pre = torch.tensor(d_t_parr[d_t_edges])
            d_t_pre = d_t_pre.to(device)
            t_t_parr = t_t_recd.cpu().detach().numpy()
            t_t_pre = torch.tensor(t_t_parr[t_t_edges])
            t_t_pre = t_t_pre.to(device)
            loss_d_t = F.binary_cross_entropy(d_t_pre, d_t_pos)
            loss_t_t = F.binary_cross_entropy(t_t_pre, t_t_pos)
            kl_divergence = 0.5 / len(train_graph.nodes()) * (
                    1 + 2 * vgae_model.log_std - vgae_model.mean ** 2 - torch.exp(vgae_model.log_std) ** 2).sum(
                1).mean()
            loss = loss_triplets + args.alpha * loss_d_t + (1 - args.alpha) * loss_t_t
            loss -= kl_divergence
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (epoch + 1) % args.evaluate_every == 0:
                vgae_model.eval()
                entity_embedding = vgae_model.sampled_z
                val_score = vgae_model.distmult(entity_embedding, val_triplets)
                val_metrics = get_eval_auc_pr_metric(val_labels, val_score)
                val_loss = F.binary_cross_entropy_with_logits(val_score, val_labels) + reg_loss
                if best_metric[1] < val_metrics[1]:
                    best_metric = val_metrics
                    counter = 0
                    torch.save({'state_dict': vgae_model.state_dict(), 'epoch': epoch + 1}, model_state_file)
                else:
                    counter += 1
                if counter > args.patience:
                    print('Early stopping!')
                    break
            if use_cuda:
                vgae_model.cuda()
        print("start testing...")
        checkpoint = torch.load(model_state_file)
        vgae_model.eval()
        vgae_model.load_state_dict(checkpoint['state_dict'])
        print("Using best epoch: {}".format(checkpoint['epoch']))
        entity_embedding = vgae_model.sampled_z
        test_score = vgae_model.distmult(entity_embedding, test_triplets)
        test_metric = get_eval_auc_pr_metric(test_labels, test_score)
        print("Fold: %d" % (fold), "test_auroc=", "{:.5f}".format(test_metric[0]), "test_aupr=","{:.5f}".format(test_metric[1]))
        final_metric += test_metric
    final_metric /= 5
    print("Final 5-cv average results", "test_auroc=", "{:.5f}".format(final_metric[0]), "test_aupr=", "{:.5f}".format(final_metric[1]))


if __name__ == '__main__':
    main()
