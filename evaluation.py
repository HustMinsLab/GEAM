import numpy as np
from munkres import Munkres, print_matrix
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from scipy.optimize import linear_sum_assignment as linear
from sklearn import metrics

def node_degree(node,array):
    degree =sum(array[node])
    return degree

def A(i,j,array):
    if array[i,j]==0:
        return 0
    else:
        return 1

def k(i,j,array):
    kij = node_degree(i,array) *node_degree(j,array)
    return kij

def judge_cluster(i,j,s,r,cluster_dict):
    if cluster_dict["layer{}".format(s+1)][i] == cluster_dict["layer{}".format(r+1)][j]:
        return 1
    else:
        return 0

def judege_fun(a,b):
    if a==b:
        return 1
    else:
        return 0

def judege_near_layer(s,r):
    if abs(s-r)==1:
        return 1
    else:
        return 0

def multi_Q(array_dict,cluster_dict,resolution = 1,coupling=1):
    q=0
    num_layer = len(array_dict)
    num_node = array_dict["layer1"].shape[0]
    u_2 = 0
    for s in range(num_layer):
        array = array_dict["layer{}".format(s+1)]
        ms=sum(sum(array))/2
        for r in range(num_layer):
            for i in range(num_node):
                for j in range(num_node):
                    if judge_cluster(i,j,s,r,cluster_dict)!=0:
                        q += (array[i,j] - resolution* (k(i,j,array)/(2*ms))) *judege_fun(s,r) + judege_fun(i,j)*(1-judege_fun(s,r))*coupling
    
    for s in range(num_layer):
        array = array_dict["layer{}".format(s+1)]
        ms=sum(sum(array))/2
        u_2 += ms
    u_2 = u_2 + num_node*((num_layer-1) *num_layer/2)* coupling
    q = q/(u_2*2)
    return q


