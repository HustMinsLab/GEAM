import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import sparse_mx_to_torch_sparse_tensor
from dataset import load


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):

        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return self.act(out)

class Readout(nn.Module):
    def __init__(self):
        super(Readout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.mean(seq * msk, 1) / torch.sum(msk)



class Attention(nn.Module):
    def __init__(self, args):
        super(Attention, self).__init__()
        self.args = args
        self.A = nn.ModuleList([nn.Linear(args.hid_units*2, 1) for _ in range(args.num_layer)])
        self.weight_init()

    def weight_init(self):
        for i in range(self.args.num_layer):
            nn.init.xavier_normal_(self.A[i].weight)
            self.A[i].bias.data.fill_(0.0)

    def forward(self, features, summary):
        
        features_attn = []
        for i in range(self.args.num_layer):
            x=torch.cat((features[i], summary[i].unsqueeze(1).expand_as(features[i])),-1)
            features_attn.append((self.A[i](torch.cat((features[i], summary[i].unsqueeze(1).expand_as(features[i])),-1))))
        features_attn = F.softmax(torch.cat(features_attn, -1), -1)
        features = torch.cat(features,1)
        
        features_attn_reshaped = features_attn.transpose(-1, -2).contiguous().view(self.args.batch_size,-1,1)
        features = features * features_attn_reshaped.expand_as(features)
        features = features.view(self.args.num_layer, self.args.batch_size, self.args.sample_size, self.args.hid_units).sum(0)
        return features


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c1, c2, h1, h2, h3, h4):
        c_x1 = torch.unsqueeze(c1, 1)
        c_x1 = c_x1.expand_as(h1).contiguous()
        c_x2 = torch.unsqueeze(c2, 1)
        c_x2 = c_x2.expand_as(h2).contiguous()

        # positive
        sc_1 = torch.squeeze(self.f_k(h2, c_x1), 2)
        sc_2 = torch.squeeze(self.f_k(h1, c_x2), 2)

        # negetive
        sc_3 = torch.squeeze(self.f_k(h4, c_x1), 2)
        sc_4 = torch.squeeze(self.f_k(h3, c_x2), 2)

        logits = torch.cat((sc_1, sc_2, sc_3, sc_4), 1)
        return logits


class Layer_Model(nn.Module):
    def __init__(self, n_in, n_h):
        super(Layer_Model, self).__init__()
        self.gcn1 = GCN(n_in, n_h)
        self.gcn2 = GCN(n_in, n_h)
        self.read = Readout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(n_h)

    def forward(self, seq1, seq2, adj, diff, sparse, msk):
        h_1 = self.gcn1(seq1, adj, sparse)
        c_1 = self.read(h_1, msk)
        c_1 = self.sigm(c_1)

        h_2 = self.gcn2(seq1, diff, sparse)
        c_2 = self.read(h_2, msk)
        c_2 = self.sigm(c_2)

        h_3 = self.gcn1(seq2, adj, sparse)
        h_4 = self.gcn2(seq2, diff, sparse)

        ret = self.disc(c_1, c_2, h_1, h_2, h_3, h_4)
        return ret, h_1, h_2, h_3, h_4, c_1, c_2

    def embed(self, seq, adj, diff, sparse, msk):
        h_1 = self.gcn1(seq, adj, sparse)
        c = self.read(h_1, msk)

        h_2 = self.gcn2(seq, diff, sparse)
        return h_1.detach(), c.detach()

class Model(nn.Module):
    def __init__(self, args, n_in, n_h=512):
        super(Model, self).__init__()
        self.args=args
        self.layers = nn.ModuleList([Layer_Model(n_in, n_h) for _ in range(args.num_layer)])
        self.disc = Discriminator(n_h)
        self.attn = Attention(args)
        self.cluster_layer = nn.Parameter(torch.Tensor(args.num_cluster, n_h))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)#Parameter中的.data属性
        self.v = args.student_t_v

    def forward(self, data, sparse, msk):
        ret=[];h_a=[];h_b=[];h_c=[];h_d=[];c_a=[];c_b=[]
        for i in range(self.args.num_layer):
            ret_s, h_as, h_bs, h_cs, h_ds, c_as, c_bs = self.layers[i](data[i][0],data[i][1],data[i][2],data[i][3], sparse, msk)
            ret.append(ret_s)
            h_a.append(h_as)
            h_b.append(h_bs)
            h_c.append(h_cs)
            h_d.append(h_ds)
            c_a.append(c_as)
            c_b.append(c_bs)  
        ret_all =  self.disc(torch.cat(c_a).mean(0).unsqueeze(0), torch.cat(c_b).mean(0).unsqueeze(0),torch.cat(h_a).mean(0).unsqueeze(0),
                        torch.cat(h_b).mean(0).unsqueeze(0), torch.cat(h_c).mean(0).unsqueeze(0), torch.cat(h_d).mean(0).unsqueeze(0))

        z = self.attn(h_a, c_a)
        z = z.squeeze(0)
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return ret, ret_all, z, q


    def embed(self, seq_list, adj_list, diff_list, sparse, msk):
        h_a=[];c_a=[]
        for i in range(self.args.num_layer):
            h_a_s,c_a_s = self.layers[0].embed(seq_list[i],adj_list[i],diff_list[i], self.args.sparse, None)
            h_a.append(h_a_s)
            c_a.append(c_a_s)
        h = self.attn(h_a, c_a).squeeze(0)
        return h.detach()
