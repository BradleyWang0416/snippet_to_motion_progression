#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math

from utils.skeleton import Skeleton
from utils.transform_initialization import Transforms


class GC4D_skip(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """

    def __init__(self, in_features, out_features, node_n=22, frame_n=35, bias=False):
        super(GC4D_skip, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.adj = Parameter(torch.zeros(frame_n, node_n, frame_n, node_n))
        self.get_adj_mask(node_n, frame_n)

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def get_adj_mask(self, node_n, frame_n):
        mask = torch.ones(frame_n, node_n, frame_n, node_n)
        for t in range(frame_n):
            mask[t, :, t, :] = 0
        self.global_pose_mask = mask.cuda()
        mask = torch.ones(frame_n, node_n, frame_n, node_n)
        for n in range(node_n):
            mask[:, n, :, n] = 0
        self.global_trajectory_mask = mask.cuda()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj_mask):
        adj_mask = adj_mask.unsqueeze(2).expand_as(self.adj)

        adj = self.adj.mul(adj_mask).mul(self.global_pose_mask).mul(self.global_trajectory_mask)

        output = torch.einsum('tvij,nijc->ntvc', adj, input)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GC4D(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """

    def __init__(self, in_features, out_features, node_n=22, frame_n=35, bias=True, mode='dynamic', neigh_norm=False):
        super(GC4D, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode
        self.neigh_norm = neigh_norm

        self.weight = Parameter(torch.FloatTensor(node_n, in_features, out_features))
        self.s = Parameter(torch.FloatTensor(frame_n, node_n, node_n))
        self.t = Parameter(torch.FloatTensor(node_n, frame_n, frame_n))

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)
        self.t.data.uniform_(-stdv, stdv)
        self.s.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def dynamic_message_passing_forward(self, x, adj_mask):
        softsign = nn.Softsign()
        x = x.mean(-1)
        x1 = x.unsqueeze(2) - x.unsqueeze(1)
        x1 = softsign(x1) / math.sqrt(self.out_features)

        x2 = x.unsqueeze(-1) - x.unsqueeze(-2)
        x2 = x2.mul(adj_mask.unsqueeze(0).expand_as(x2))
        x2 = softsign(x2) / math.sqrt(self.out_features)

        return x1.permute(0, 3, 1, 2), x2

    def neighborhood_normalize_forward(self, adj_mask):
        num_node = adj_mask.shape[-1]
        d = torch.sum(adj_mask[0], 1)
        deg = torch.zeros(num_node, num_node)
        for i in range(num_node):
            if d[i] > 0:
                deg[i, i] = d[i] ** (-1)
        return torch.einsum('vi,tiu->tvu', deg.cuda(), self.s.mul(adj_mask))


    def forward(self, input, adj_mask):
        # 输入[N,T,V,C]，第一层输入为[N,35,22,3]

        if self.neigh_norm:
            s = self.neighborhood_normalize_forward(adj_mask)
        else:
            s = self.s.mul(adj_mask)

        if self.mode == 'dynamic':
            tt, ss = self.dynamic_message_passing_forward(input, adj_mask)
            aggreg_s = torch.einsum('ntvi,ntic->ntvc', s.unsqueeze(0) + ss, input)
            aggreg_t = torch.einsum('nvtj,njvc->ntvc', self.t.unsqueeze(0) + tt, input)
        else:
            aggreg_s = torch.einsum('tvi,ntic->ntvc', s, input)
            aggreg_t = torch.einsum('vtj,njvc->ntvc', self.t, input)

        support = aggreg_s + aggreg_t

        output = torch.einsum('ntvj,vjc->ntvc', support, self.weight)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class Encode_Block(nn.Module):
    def __init__(self, in_features, out_features, p_dropout, node_n=22, frame_n=35, bias=True, mode='dynamic', neigh_norm=False):
        """
        Define a residual block of GCN
        """
        super(Encode_Block, self).__init__()

        self.gc1 = GC4D(in_features, out_features, node_n=node_n, frame_n=frame_n, bias=bias, mode=mode, neigh_norm=neigh_norm)
        self.bn1 = nn.BatchNorm2d(out_features)
        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x, adj_mask):
        y = self.gc1(x, adj_mask)
        y = self.bn1(y.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        y = self.act_f(y)
        y = self.do(y)

        return y

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GC4D_Block(nn.Module):
    def __init__(self, in_features, p_dropout, node_n=22, frame_n=35, bias=True, mode='dynamic', neigh_norm=False):
        super(GC4D_Block, self).__init__()
        self.gc1 = GC4D(in_features, in_features, node_n=node_n, frame_n=frame_n, bias=bias, mode=mode, neigh_norm=neigh_norm)
        self.bn1 = nn.BatchNorm2d(in_features)

        self.gc2 = GC4D(in_features, in_features, node_n=node_n, frame_n=frame_n, bias=bias, mode=mode, neigh_norm=neigh_norm)
        self.bn2 = nn.BatchNorm2d(in_features)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x, adj_mask):
        y = self.gc1(x, adj_mask)

        y = self.bn1(y.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y, adj_mask)

        y = self.bn2(y.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        y = self.act_f(y)
        y = self.do(y)

        return y + x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


def construct_graph(x, transform):
    return torch.einsum('btvc,vn->btnc', x, transform)


class DDGCN(nn.Module):
    def __init__(self, input_feature, hidden_feature, p_dropout, num_block=3, joint_n=[22, 11, 2], frame_n=60, skl_type='h36m'):
        super(DDGCN, self).__init__()

        joint_n = [22, 11]


        self.num_block = num_block
        self.joint_n = joint_n

        self.adjacency_mask = []
        self.get_adjacency_mask(skl_type, joint_n, frame_n)

        self.transform = nn.ParameterList()
        self.get_skeleton_transform(joint_n)

        # scale 0
        self.gc_en = Encode_Block(in_features=input_feature, out_features=hidden_feature, p_dropout=p_dropout, node_n=joint_n[0], frame_n=frame_n, mode='static', neigh_norm=True)
        self.gcn = nn.ModuleList()
        for _ in range(num_block):
            self.gcn.append(GC4D_Block(hidden_feature, p_dropout=p_dropout, node_n=joint_n[0], frame_n=frame_n))
        self.gc_de = GC4D(in_features=hidden_feature, out_features=input_feature, node_n=joint_n[0], frame_n=frame_n)

        self.gc_skip = GC4D_skip(in_features=hidden_feature, out_features=hidden_feature, node_n=joint_n[0], frame_n=frame_n)

        # scale 1
        self.gc_en_s1 = Encode_Block(in_features=input_feature, out_features=hidden_feature, p_dropout=p_dropout, node_n=joint_n[1], frame_n=frame_n, mode='static', neigh_norm=True)
        self.gcn_s1 = nn.ModuleList()
        for _ in range(num_block):
            self.gcn_s1.append(GC4D_Block(hidden_feature, p_dropout=p_dropout, node_n=joint_n[1], frame_n=frame_n))
        self.gc_de_s1 = GC4D(in_features=hidden_feature, out_features=input_feature, node_n=joint_n[1], frame_n=frame_n)


        # inv_skeleton transformation
        inv_transform_s1 = torch.zeros(joint_n[1], joint_n[0])
        inv_transform_s1[Transforms['M{}to{}'.format(joint_n[0], joint_n[1])].permute(1, 0) != 0] = 1
        self.inv_transform_s1 = nn.ParameterList([Parameter(inv_transform_s1) for _ in range(self.num_block + 2)])


    def get_adjacency_mask(self, skl_type, joint_n, frame_n):
        for num_joint in joint_n:
            skl = Skeleton(skl_type, num_joint).skeleton
            skl = torch.tensor(skl, dtype=torch.float32, requires_grad=False)
            bi_skl = torch.zeros(num_joint, num_joint)
            bi_skl[skl != 0] = 1.
            bi_skl = bi_skl.unsqueeze(0).expand(frame_n, -1, -1)
            self.adjacency_mask.append(bi_skl.cuda())

    def get_skeleton_transform(self, joint_n):
        for num_joint in joint_n:
            trans = Transforms['M{}to{}'.format(joint_n[0], num_joint)]
            if joint_n[0] == num_joint:
                self.transform.append(Parameter(trans, requires_grad=False))
            else:
                self.transform.append(Parameter(trans))

    def forward(self, input):


        x_init = [input for _ in self.joint_n]

        x = list(map(construct_graph, x_init, self.transform))

        # encode
        y0 = self.gc_en(x[0], self.adjacency_mask[0])
        y1 = self.gc_en_s1(x[1], self.adjacency_mask[1])

        y_skip = self.gc_skip(y0, self.adjacency_mask[0])
        y0 = y0 + torch.einsum('btvc,vn->btnc', y1, self.inv_transform_s1[0])
        for i in range(self.num_block):
            y0 = self.gcn[i](y0, self.adjacency_mask[0])
            y1 = self.gcn_s1[i](y1, self.adjacency_mask[1])
            y0 = y0 + torch.einsum('btvc,vn->btnc', y1, self.inv_transform_s1[i + 1])
        y0 = y0 + y_skip

        # decode
        y0 = self.gc_de(y0, self.adjacency_mask[0])
        y1 = self.gc_de_s1(y1, self.adjacency_mask[1])
        y0 = y0 + torch.einsum('btvc,vn->btnc', y1, self.inv_transform_s1[-1])

        return y0 + x[0]
