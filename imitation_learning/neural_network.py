import dgl.function as fn
import numpy as np
import torch
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from dgl.nn.functional import edge_softmax
from torch import nn
from torch.nn import init


class EGATConv(nn.Module):
    r"""

    Description
    -----------
    Apply Graph Attention Layer over input graph. EGAT is an extension
    of regular `Graph Attention Network <https://arxiv.org/pdf/1710.10903.pdf>`__
    handling edge features, detailed description is available in
    `Rossmann-Toolbox <https://pubmed.ncbi.nlm.nih.gov/34571541/>`__ (see supplementary data).
     The difference appears in the method how unnormalized attention scores :math:`e_{ij}`
     are obtain:

    .. math::
        e_{ij} &= \vec{F} (f_{ij}^{\prime})
        f_{ij}^{\prim} &= \mathrm{LeakyReLU}\left(A [ h_{i} \| f_{ij} \| h_{j}]\right)
    where :math:`f_{ij}^{\prim}` are edge features, :math:`\mathrm{A}` is weight matrix and
    :math: `\vec{F}` is weight vector. After that resulting node features
    :math:`h_{i}^{\prim}` are updated in the same way as in regular GAT.

    Parameters
    ----------
    in_node_feats : int
        Input node feature size :math:`h_{i}`.
    in_edge_feats : int
        Input edge feature size :math:`f_{ij}`.
    out_node_feats : int
        Output nodes feature size.
    out_edge_feats : int
        Output edge feature size.
    num_heads : int
        Number of attention heads.
    bias : bool, optional
        If True, learns a bias term. Defaults: ``True``.

    Examples
    ----------
    # >>> import dgl
    # >>> import torch as th
    # >>> from dgl.nn import EGATConv
    # >>>
    # >>> num_nodes, num_edges = 8, 30
    # >>>#define connections
    # >>> u, v = th.randint(num_nodes, num_edges), th.randint(num_nodes, num_edges)
    # >>> graph = dgl.graph((u,v))
    # >>> node_feats = th.rand((num_nodes, 20))
    # >>> edge_feats = th.rand((num_edges, 12))
    # >>> egat = EGATConv(in_node_feats=20,
                          in_edge_feats=12,
                          out_node_feats=15,
                          out_edge_feats=10,
                          num_heads=3)
    # >>> #forward pass
    # >>> new_node_feats, new_edge_feats = egat(graph, node_feats, edge_feats)
    # >>> new_node_feats.shape, new_edge_feats.shape
    ((8, 3, 12), (30, 3, 10))
    """

    def __init__(self,
                 in_node_feats,
                 in_edge_feats,
                 out_node_feats,
                 out_edge_feats,
                 num_heads,
                 bias=True,
                 **kw_args):

        super().__init__()
        self._num_heads = num_heads
        self._out_node_feats = out_node_feats
        self._out_edge_feats = out_edge_feats
        self.fc_node = nn.Linear(in_node_feats, out_node_feats * num_heads, bias=True)
        self.fc_ni = nn.Linear(in_node_feats, out_edge_feats * num_heads, bias=False)
        self.fc_fij = nn.Linear(in_edge_feats, out_edge_feats * num_heads, bias=False)
        self.fc_nj = nn.Linear(in_node_feats, out_edge_feats * num_heads, bias=False)
        self.attn = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_edge_feats)))
        if bias:
            self.bias = nn.Parameter(th.FloatTensor(size=(num_heads * out_edge_feats,)))
        else:
            self.register_buffer('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reinitialize learnable parameters.
        """
        gain = init.calculate_gain('relu')
        init.xavier_normal_(self.fc_node.weight, gain=gain)
        init.xavier_normal_(self.fc_ni.weight, gain=gain)
        init.xavier_normal_(self.fc_fij.weight, gain=gain)
        init.xavier_normal_(self.fc_nj.weight, gain=gain)
        init.xavier_normal_(self.attn, gain=gain)
        nn.init.constant_(self.bias, 0)

    def forward(self, graph, nfeats, efeats, get_attention=False):
        r"""
        Compute new node and edge features.
        Parameters
        ----------
        graph : DGLGraph
            The graph.
        nfeats : torch.Tensor
            The input node feature of shape :math:`(*, D_{in})`
            where:
                :math:`D_{in}` is size of input node feature,
                :math:`*` is the number of nodes.
        efeats: torch.Tensor
             The input edge feature of shape :math:`(*, F_{in})`
             where:
                 :math:`F_{in}` is size of input node feauture,
                 :math:`*` is the number of edges.
        get_attention : bool, optional
                Whether to return the attention values. Default to False.

        Returns
        -------
        pair of torch.Tensor
            node output features followed by edge output features
            The node output feature of shape :math:`(*, H, D_{out})`
            The edge output feature of shape :math:`(*, H, F_{out})`
            where:
                :math:`H` is the number of heads,
                :math:`D_{out}` is size of output node feature,
                :math:`F_{out}` is size of output edge feature.
        """

        with graph.local_scope():
            # TODO allow node src and dst feats
            graph.edata['f'] = efeats
            graph.ndata['h'] = nfeats
            # calc edge attention
            # same trick way as in dgl.nn.pytorch.GATConv, but also includes edge feats
            # https://github.com/dmlc/dgl/blob/master/python/dgl/nn/pytorch/conv/gatconv.py#L297
            f_ni = self.fc_ni(nfeats)
            f_nj = self.fc_nj(nfeats)
            f_fij = self.fc_fij(efeats)
            graph.srcdata.update({'f_ni': f_ni})
            graph.dstdata.update({'f_nj': f_nj})
            # graph.edata.update({'f_fij' : f_fij})
            # add ni, nj factors
            graph.apply_edges(fn.u_add_v('f_ni', 'f_nj', 'f_tmp'))
            # add fij to node factor
            f_out = graph.edata.pop('f_tmp') + f_fij
            if self.bias is not None:
                f_out += self.bias
            f_out = nn.functional.leaky_relu(f_out)
            f_out = f_out.view(-1, self._num_heads, self._out_edge_feats)
            # compute attention factor
            e = (f_out * self.attn).sum(dim=-1).unsqueeze(-1)
            graph.edata['a'] = edge_softmax(graph, e)
            graph.ndata['h_out'] = self.fc_node(nfeats).view(-1, self._num_heads, self._out_node_feats)
            # calc weighted sum
            graph.update_all(fn.u_mul_e('h_out', 'a', 'm'),
                             fn.sum('m', 'h_out'))

            h_out = graph.ndata['h_out'].view(-1, self._num_heads, self._out_node_feats)
            if get_attention:
                return h_out, f_out, graph.edata.pop('a')
            else:
                return h_out, f_out


class BoundPredictor(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 300)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(300, 300)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(300, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


class RemovalTimePredictor(nn.Module):
    """
    Network architecture based on the GAT architecture with edge features. This architecture also used multiple
    GAT layers which is why I used it here together with normalization and nonlinearities. They recommended ReLU,
    batchNorm is something i came up with myself (which is probably not useful since batch_size = 1 for us)
    """

    def __init__(self):  # TODO find out out_features optimal sizes
        super().__init__()
        self.relu1 = nn.ReLU()
        def gnn_block(inputs, outputs, heads):
            egat = EGATConv(*inputs, *outputs, heads, bias=True)
            output_shape = np.array(outputs)*heads
            return egat, output_shape
        edge_features = 15
        node_features = 15
        heads = 60
        self.layers = []
        egat, self.output_shape1 = gnn_block(inputs=(3,1), outputs=(node_features,edge_features), heads=heads)
        self.layers.append(egat)
        layers = 7
        output_shape = self.output_shape1
        for i in range(layers):
            egat, output_shape = gnn_block(inputs=output_shape, outputs=(node_features,edge_features), heads=heads)
            self.layers.append(egat)
        egat_final, _ = gnn_block(inputs=output_shape, outputs=(1, 1), heads=1)
        self.layers.append(egat_final)
        # self.all_parameters = nn.ParameterList([l.parameters() for l in self.layers])
        self.module_list = nn.ModuleList(self.layers)

    def _forward_layer(self, graph, node_f, edge_f, gnn):
        new_node_feats, new_edge_feats = gnn(graph, node_f, edge_f)  # output shape: N x Heads x out_feats
        new_node_feats, new_edge_feats = self.relu1(new_node_feats), self.relu1(new_edge_feats)
        # we flatten the multi heads into 1 dimension such that we can apply the GAT again:
        nodes, edges = self.output_shape1
        new_node_feats = new_node_feats.reshape(-1, nodes)
        new_edge_feats = new_edge_feats.reshape(-1, edges)
        return new_node_feats, new_edge_feats

    def forward(self, graph, node_f, edge_f):
        new_node_feats, new_edge_feats = node_f, edge_f
        for egat in self.layers[:-1]:
            new_node_feats, new_edge_feats = self._forward_layer(graph, new_node_feats, new_edge_feats, egat)
        new_node_feats, new_edge_feats = self.layers[-1](graph, new_node_feats, new_edge_feats)
        return new_node_feats, new_edge_feats