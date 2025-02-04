"""
Baseline GNN models
"""

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Linear, Parameter, BatchNorm1d as BN
from torch_sparse import SparseTensor
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.nn.dense.linear import Linear as pygLinear
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.nn.inits import zeros
import torch_sparse
from torch import nn

class Decoder(nn.Module):
    def __init__(self, dim_z, dim_sf, cuda, dim_h=64):
        super(Decoder, self).__init__()
        dim_in = dim_z
        self.mlp_embed = nn.Sequential(
            nn.Linear(dim_in, dim_in, bias=True),
            nn.BatchNorm1d(dim_in),
            nn.ReLU(),
            nn.Dropout(p=0.3),
        )
        concat_dim = dim_in
        self.out_1 = nn.Sequential(nn.Linear(concat_dim, 1, bias=False), nn.Sigmoid())
        self.out_0 = nn.Sequential(nn.Linear(concat_dim, 1, bias=False), nn.Sigmoid())
        self.lin = Linear(concat_dim, 1)


    def forward(self, z, T):
        # z is the subgraph feature
        z = z[:, 0, :] * z[:, 1, :]
        z = self.mlp_embed(z)
        h0 = z
        h_1 = self.out_1(h0).squeeze()  # (N,)
        h_0 = self.out_0(h0).squeeze()
        t1 = T.squeeze().float()
        res_f = torch.zeros_like(t1).to(T.device)
        # return self.lin(h0)
        res_f[t1 == 0] = h_0[t1 == 0]
        res_f[t1 == 1] = h_1[t1 == 1]
        h = res_f
        return h

    def reset_parameters(self):
        for lin in self.mlp_out:
            try:
                lin.reset_parameters()
            except:
                continue


class GCN(torch.nn.Module):
    def __init__(self, args, num_feature, num_layers,dropout=0.3):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        dim_h = args.hidden_channels
        self.convs.append(GCNConv(num_feature, dim_h, cached=True))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(dim_h, dim_h, cached=True))
        self.convs.append(GCNConv(dim_h, dim_h, cached=True))
        if args.dblp:
            self.predictor = Decoder(args.hidden_channels, args.max_hash_hops * (args.max_hash_hops + 2),'cuda:0')
        else:
            self.predictor = LinkPredictor(dim_h,dim_h,1,2,0.3)
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class GCNCustomConv(torch.nn.Module):
    """
    Class to propagate features
    """

    def __init__(self, in_channels, out_channels, bias=True, propagate_features=False, **kwargs):
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self._cached_edge_index = None
        self._cached_adj_t = None
        self.propagate_features = propagate_features

        self.lin = pygLinear(in_channels, out_channels, bias=False,
                             weight_initializer='glorot')

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        # do the XW bit first
        x = self.lin(x)
        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        edge_index, edge_weight = gcn_norm(  # yapf: disable
            edge_index, edge_weight, x.size(0))
        if self.propagate_features:
            out = torch_sparse.spmm(edge_index, edge_weight, x.shape[0], x.shape[0], x)
        else:
            out = x
        if self.bias is not None:
            out += self.bias
        return out

    def reset_parameters(self):
        self.lin.reset_parameters()
        zeros(self.bias)


class SAGE(torch.nn.Module):
    def __init__(self, args, num_feature, num_layers,
                 dropout=0.3, residual=True):
        super(SAGE, self).__init__()
        dim_h = args.hidden_channels
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(num_feature, dim_h, root_weight=residual))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(dim_h, dim_h, root_weight=residual))
        self.convs.append(SAGEConv(dim_h, dim_h, root_weight=residual))
        if args.dblp:
            self.predictor = Decoder(args.hidden_channels, args.max_hash_hops * (args.max_hash_hops + 2),'cuda:0')
        else:
            self.predictor = LinkPredictor(dim_h,dim_h,1,2,0.3)

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class SIGNBaseClass(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, K, dropout):
        super(SIGNBaseClass, self).__init__()

        self.K = K
        self.lins = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for _ in range(self.K + 1):
            self.lins.append(Linear(in_channels, hidden_channels))
            self.bns.append(BN(hidden_channels))
        self.lin_out = Linear((K + 1) * hidden_channels, out_channels)
        self.dropout = dropout
        self.adj_t = None

    def reset_parameters(self):
        for lin, bn in zip(self.lins, self.bns):
            lin.reset_parameters()
            bn.reset_parameters()

    def cache_adj_t(self, edge_index, num_nodes):
        row, col = edge_index
        adj_t = SparseTensor(row=col, col=row,
                             sparse_sizes=(num_nodes, num_nodes))

        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        return deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)

    def forward(self, *args):
        raise NotImplementedError


class SIGNEmbedding(SIGNBaseClass):
    def __init__(self, in_channels, hidden_channels, out_channels, K, dropout):
        super(SIGNEmbedding, self).__init__(in_channels, hidden_channels, out_channels, K, dropout)

    def forward(self, x, adj_t, num_nodes):
        if self.adj_t is None:
            self.adj_t = self.cache_adj_t(adj_t, num_nodes)
        hs = []
        for lin, bn in zip(self.lins, self.bns):
            h = lin(x)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            hs.append(h)
            x = self.adj_t @ x
        h = torch.cat(hs, dim=-1)
        x = self.lin_out(h)
        return x


class SIGN(SIGNBaseClass):
    def __init__(self, in_channels, hidden_channels, out_channels, K, dropout):
        super(SIGN, self).__init__(in_channels, hidden_channels, out_channels, K, dropout)

    def forward(self, xs):
        """
        apply the sign feature transform where each component of the polynomial A^n x is treated independently
        @param xs: [batch_size, 2, n_features * (K + 1)]
        @return: [batch_size, 2, hidden_dim]
        """
        xs = torch.tensor_split(xs, self.K + 1, dim=-1)
        hs = []
        # split features into k+1 chunks and put each tensor in a list
        for lin, bn, x in zip(self.lins, self.bns, xs):
            h = lin(x)
            # the next line is a fuggly way to apply the same batch norm to both source and destination edges
            h = torch.cat((bn(h[:, 0, :]).unsqueeze(1), bn(h[:, 1, :]).unsqueeze(1)), dim=1)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            hs.append(h)
        h = torch.cat(hs, dim=-1)
        x = self.lin_out(h)
        return x


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, z):
        x = z[:, 0, :] * z[:, 1, :]

        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)
