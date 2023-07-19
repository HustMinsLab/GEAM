from operator import index
import re
from networkx.linalg.modularitymatrix import modularity_matrix
import numpy as np
from numpy import ma
from numpy.core.records import array
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import *
from utils import sparse_mx_to_torch_sparse_tensor
from dataset import load, load_graph
import time

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn import metrics
from evaluation import *


def train(args, verbose=True):

    num_layer=args.num_layer
    nb_epochs = args.nb_epochs
    patience = args.patience
    lr = args.lr
    l2_coef = args.l2_coef
    hid_units = args.hid_units
    sparse = args.sparse

    dataset = args.dataset
  
    _, _, features, labels = load(dataset,1,args)

    ft_size = features.shape[1]
    nb_classes = np.unique(labels).shape[0]

    sample_size = args.sample_size
    batch_size = args.batch_size
    labels = torch.LongTensor(labels)
    lbl_1 = torch.ones(batch_size, sample_size * 2)
    lbl_2 = torch.zeros(batch_size, sample_size * 2)
    lbl = torch.cat((lbl_1, lbl_2), 1)
    model = Model(args, ft_size, hid_units)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
    
    model.cpu()
    labels = labels.cpu()
    lbl = lbl.cpu()

    b_xent = nn.BCEWithLogitsLoss()
    xent = nn.CrossEntropyLoss()
    
    time_before_train=time.time()

    data=[]
    
    for epoch in range(nb_epochs):
        time_before_epoch=time.time()
        for layer in range(num_layer):
                    
            adj, diff, features, labels = load(dataset,layer+1,args)
            ba, bd, bf = [], [], []
            i=0
            # adj
            ba.append(adj[i: i + sample_size, i: i + sample_size])
            # diffuse
            bd.append(diff[i: i + sample_size, i: i + sample_size])
            # feature
            bf.append(features[i: i + sample_size])

            ba = np.array(ba).reshape(batch_size, sample_size, sample_size)
            bd = np.array(bd).reshape(batch_size, sample_size, sample_size)
            bf = np.array(bf).reshape(batch_size, sample_size, ft_size)

            if sparse:
                ba = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(ba))
                bd = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(bd))
            else:
                ba = torch.FloatTensor(ba)
                bd = torch.FloatTensor(bd)

            bf = torch.FloatTensor(bf)
            idx = np.random.permutation(sample_size)
            shuf_fts = bf[:, idx, :]

            bf = bf.cpu()
            ba = ba.cpu()
            bd = bd.cpu()
            shuf_fts = shuf_fts.cpu()

            data.append([bf, shuf_fts, ba, bd])
        
        model.train()
        optimiser.zero_grad()
        
        logits, logit_all, z, q = model(data, sparse, None) 

        p = target_distribution(q)
        loss_cl=0
        for i in range(num_layer):
            loss_cl += b_xent(logits[i], lbl)

        loss_cl_all =  b_xent(logit_all, lbl)
        loss_kl = F.kl_div(q.log(), p, reduction='batchmean')
        loss = loss_cl
        print("\nEpoch: {}  loss: {}".format(epoch,loss))
    
        H = get_attn_embed(args,model,verbose=True,sparse=False)
        n,a,p,Q = test(args,H,labels,epoch,verbose) 


        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        time_after_epoch=time.time()
        print("Cost time: {:.2f}".format(time_after_epoch-time_before_epoch))


def test(args,H,labels,epoch,verbose):
    Q=0;n=0;a=0;p=0
    cal = KMeans(n_clusters=args.num_cluster, n_init=20)
    y_pred = cal.fit_predict(H.cpu().numpy())

    if args.with_gt and epoch%args.perEpoch_Q==0 and epoch!=0:
        n=nmi_score(labels,y_pred)
        a=ari_score(labels,y_pred)
        p=purity_score(labels,y_pred)
        if verbose:
            print("NMI: {:.4f}".format(n)," ARI: {:.4f}".format(a)," Purity: {:.4f}".format(p))
            
    if args.test_Q and epoch%args.perEpoch_Q==0 and epoch!=0: 
        array_dict={"layer{}".format(i+1):load_graph(args.dataset,i+1) for i in range(args.num_layer)}
        cluster_dict={"layer{}".format(i+1):y_pred for i in range(args.num_layer)}
        Q=multi_Q(array_dict, cluster_dict)
        print("Modarility: {:.4f}".format(Q))
    return n,a,p,Q


def get_attn_embed(args,model,verbose=True,sparse=False):
    features_list=[];adj_list=[];diff_list=[]
    for layer in range(args.num_layer):
        adj_s, diff_s, features_s, labels = load(args.dataset,layer+1,args)
        
        features_s = torch.FloatTensor(features_s[np.newaxis])
        adj_s = torch.FloatTensor(adj_s[np.newaxis])
        diff_s = torch.FloatTensor(diff_s[np.newaxis])
        
        features_list.append(features_s)
        adj_list.append(adj_s)
        diff_list.append(diff_s)
        
        H_s,_ = model.layers[layer].embed(features_s,adj_s,diff_s, sparse, None)
    H = model.embed(features_list, adj_list, diff_list, sparse, None)
    return H



def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 



