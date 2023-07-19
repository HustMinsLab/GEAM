from dgl.data import CoraDataset, CitationGraphDataset
from utils import preprocess_features, normalize_adj,normalize
from sklearn.preprocessing import MinMaxScaler
from utils import *
import scipy.sparse as sp
import networkx as nx
import numpy as np
import os

import torch
from torch.utils.data import Dataset, dataset



def load(dataset,layer_id,args):
    diffusion=args.diffusion
    initial_feat=args.initial_feat
    datadir = os.path.join('data', dataset, diffusion, initial_feat, "layer"+str(layer_id))
    
    if not os.path.exists(datadir):
       
        adj = load_graph(dataset,layer_id)
        if diffusion=="ppr":
            diff = compute_ppr(adj, 0.2)
        elif diffusion=="heat":
            diff = compute_heat(adj,5)
        
        if initial_feat=="deepwalk":
            feat = np.loadtxt('multi_data/{}_layer{}.txt'.format(dataset, layer_id), dtype=float)
        elif initial_feat=="node2vec":
            feat = np.loadtxt('multi_data/{}_layer{}_node2vec.txt'.format(dataset, layer_id), dtype=float)

        labels = np.loadtxt('multi_data/{}_label.txt'.format(dataset), dtype=int)

        os.makedirs(datadir)
        np.save('{}/adj.npy'.format(datadir), adj)
        np.save('{}/diff.npy'.format(datadir), diff)
        np.save('{}/feat.npy'.format(datadir), feat)
        np.save('{}/labels.npy'.format(datadir), labels)
       
    else:
        adj = np.load('{}/adj.npy'.format(datadir))
        diff = np.load('{}/diff.npy'.format(datadir))
        feat = np.load('{}/feat.npy'.format(datadir))
        labels = np.load('{}/labels.npy'.format(datadir))
        
    adj = normalize_adj(adj + sp.eye(adj.shape[0])).todense()

    return adj, diff, feat, labels

def check_symmetric(a, tol=1e-8):
    return np.all(np.abs(a-a.T) < tol)

def load_graph(dataset, layer_id):
    path = 'multi_graph/{}_graph_layer{}.txt'.format(dataset,layer_id)
    data = np.loadtxt('multi_data/{}_layer{}.txt'.format(dataset,layer_id))
    n, _ = data.shape
    
    if 'aucs' in dataset:
        idx=[]
        lis = np.loadtxt('multi_data/{}_label.txt'.format(dataset),delimiter=' ', dtype=int)
        for i in range(lis.shape[0]):
            idx.append(lis[i][0])
        idx = np.array(idx)
    else:
         idx = np.array([i for i in range(1,n+1)], dtype=np.int32)

    edges_unordered = np.genfromtxt(path, dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                    dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(n, n), dtype=np.float64)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj_m =adj.todense()
    try:
        if(not check_symmetric(adj_m)):
            raise ValueError("adj not sym")
    except ValueError as e:
        print("error：",repr(e))
    adj_m =np.array(adj_m)
    return adj_m

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx