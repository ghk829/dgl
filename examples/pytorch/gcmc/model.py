"""NN modules"""
import torch as th
import torch.nn as nn
from torch.nn import init
import dgl.function as fn
import dgl.nn.pytorch as dglnn
from dgl.nn.pytorch.conv import GATConv
from utils import get_activation, to_etype_name

class GCMCGraphConv(nn.Module):
    """Graph convolution module used in the GCMC model.

    Parameters
    ----------
    in_feats : int
        Input feature size.
    out_feats : int
        Output feature size.
    weight : bool, optional
        If True, apply a linear layer. Otherwise, aggregating the messages
        without a weight matrix or with an shared weight provided by caller.
    device: str, optional
        Which device to put data in. Useful in mix_cpu_gpu training and
        multi-gpu training
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 weight=True,
                 device=None,
                 dropout_rate=0.0):
        super(GCMCGraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.device = device
        self.dropout = nn.Dropout(dropout_rate)

        if weight:
            self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        if self.weight is not None:
            init.xavier_uniform_(self.weight)

    def forward(self, graph, feat, weight=None):
        """Compute graph convolution.

        Normalizer constant :math:`c_{ij}` is stored as two node data "ci"
        and "cj".

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature
        weight : torch.Tensor, optional
            Optional external weight tensor.
        dropout : torch.nn.Dropout, optional
            Optional external dropout layer.

        Returns
        -------
        torch.Tensor
            The output feature
        """
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat, _ = feat      # dst feature not used
            cj = graph.srcdata['cj']
            ci = graph.dstdata['ci']
            if self.device is not None:
                cj = cj.to(self.device)
                ci = ci.to(self.device)
            if weight is not None:
                if self.weight is not None:
                    raise DGLError('External weight is provided while at the same time the'
                                   ' module has defined its own weight parameter. Please'
                                   ' create the module with flag weight=False.')
            else:
                weight = self.weight

            if weight is not None:
                feat = dot_or_identity(feat, weight, self.device)

            feat = feat * self.dropout(cj)
            graph.srcdata['h'] = feat
            graph.update_all(fn.copy_src(src='h', out='m'),
                             fn.sum(msg='m', out='h'))
            rst = graph.dstdata['h']
            rst = rst * ci

        return rst


class GCMCGraphGAT(nn.Module):
    """Graph convolution module used in the GCMC model.

    Parameters
    ----------
    in_feats : int
        Input feature size.
    out_feats : int
        Output feature size.
    weight : bool, optional
        If True, apply a linear layer. Otherwise, aggregating the messages
        without a weight matrix or with an shared weight provided by caller.
    device: str, optional
        Which device to put data in. Useful in mix_cpu_gpu training and
        multi-gpu training
    """
    def __init__(self,
                 in_feats:tuple,
                 out_feats,
                 weight=True,
                 device=None,
                 dropout_rate=0.0):
        super(GCMCGraphGAT, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.device = device
        self.dropout = nn.Dropout(dropout_rate)

        if weight:
            self.feat1 = nn.Parameter(th.Tensor(in_feats[0], 10))
            self.feat2 = nn.Parameter(th.Tensor(in_feats[1], 10))
        else:
            self.register_parameter('weight', None)
        self.reset_parameters()

        self.gat = GATConv((10,10), out_feats, 8, attn_drop=dropout_rate, allow_zero_in_degree=True)

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        init.xavier_uniform_(self.feat1)
        init.xavier_uniform_(self.feat2)

    def forward(self, graph, feat, weight=None):
        """Compute graph convolution.

        Normalizer constant :math:`c_{ij}` is stored as two node data "ci"
        and "cj".

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature
        weight : torch.Tensor, optional
            Optional external weight tensor.
        dropout : torch.nn.Dropout, optional
            Optional external dropout layer.

        Returns
        -------
        torch.Tensor
            The output feature
        """

        return self.gat(graph, (self.feat1, self.feat2))


class GCMCGraphSage(nn.Module):
    """Graph convolution module used in the GCMC model.

    Parameters
    ----------
    in_feats : int
        Input feature size.
    out_feats : int
        Output feature size.
    weight : bool, optional
        If True, apply a linear layer. Otherwise, aggregating the messages
        without a weight matrix or with an shared weight provided by caller.
    device: str, optional
        Which device to put data in. Useful in mix_cpu_gpu training and
        multi-gpu training
    """
    def __init__(self,
                 in_feats:tuple,
                 out_feats,
                 weight=True,
                 device=None,
                 dropout_rate=0.0):
        super(GCMCGraphSage, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.device = device
        self.dropout = nn.Dropout(dropout_rate)

        if weight:
            self.feat1 = nn.Parameter(th.Tensor(in_feats[0], 10))
            self.feat2 = nn.Parameter(th.Tensor(in_feats[1], 10))
        else:
            self.register_parameter('weight', None)
        self.reset_parameters()
        from dgl.nn.pytorch.conv import SAGEConv
        self.sage = SAGEConv(10, out_feats, aggregator_type='mean')

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        init.xavier_uniform_(self.feat1)
        init.xavier_uniform_(self.feat2)

    def forward(self, graph, feat, weight=None):

        return self.sage(graph, (self.feat1, self.feat2))


class GraphSageLayer(nn.Module):
    def __init__(self,
                 rating_vals,
                 user_in_units,
                 movie_in_units,
                 msg_units,
                 out_units,
                 dropout_rate=0.0,
                 agg='stack',  # or 'sum'
                 agg_act=None,
                 out_act=None,
                 share_user_item_param=False,
                 device=None):
        super(GraphSageLayer, self).__init__()
        self.rating_vals = rating_vals
        self.agg = agg
        self.share_user_item_param = share_user_item_param
        self.ufc = nn.Linear(msg_units, out_units)
        if share_user_item_param:
            self.ifc = self.ufc
        else:
            self.ifc = nn.Linear(msg_units, out_units)
        if agg == 'stack':
            # divide the original msg unit size by number of ratings to keep
            # the dimensionality
            assert msg_units % len(rating_vals) == 0
            msg_units = msg_units // len(rating_vals)
        self.dropout = nn.Dropout(dropout_rate)
        self.W_r = nn.ParameterDict()
        subConv = {}
        subConv2 = {}
        for rating in rating_vals:
            # PyTorch parameter name can't contain "."
            rating = to_etype_name(rating)
            rev_rating = 'rev-%s' % rating
            if share_user_item_param and user_in_units == movie_in_units:
                self.W_r[rating] = nn.Parameter(th.randn(user_in_units, msg_units))
                self.W_r['rev-%s' % rating] = self.W_r[rating]
                subConv[rating] = GCMCGraphGAT(user_in_units,
                                               msg_units,
                                               weight=False,
                                               device=device,
                                               dropout_rate=dropout_rate)
                subConv[rev_rating] = GCMCGraphGAT(user_in_units,
                                                   msg_units,
                                                   weight=False,
                                                   device=device,
                                                   dropout_rate=dropout_rate)
            else:
                self.W_r = None
                subConv[rating] = GCMCGraphSage((user_in_units,movie_in_units),
                                                msg_units,
                                                weight=True,
                                                device=device,
                                                dropout_rate=dropout_rate)
                subConv[rev_rating] = GCMCGraphSage((movie_in_units,user_in_units),
                                                    msg_units,
                                                    weight=True,
                                                    device=device,
                                                    dropout_rate=dropout_rate)

        self.conv = dglnn.HeteroGraphConv(subConv, aggregate=agg)
        # self.conv2 = dglnn.HeteroGraphConv(subConv2, aggregate=agg)
        self.agg_act = get_activation(agg_act)
        self.out_act = get_activation(out_act)
        self.device = device
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, graph, ufeat=None, ifeat=None):
        in_feats = {'user' : ufeat, 'item' : ifeat}
        mod_args = {}
        for i, rating in enumerate(self.rating_vals):
            rating = to_etype_name(rating)
            rev_rating = 'rev-%s' % rating
            mod_args[rating] = (self.W_r[rating] if self.W_r is not None else None,)
            mod_args[rev_rating] = (self.W_r[rev_rating] if self.W_r is not None else None,)
        out_feats = self.conv(graph, in_feats, mod_args=mod_args)

        ufeat = out_feats['user']
        ifeat = out_feats['item']
        ufeat = ufeat.view(ufeat.shape[0], -1)
        ifeat = ifeat.view(ifeat.shape[0], -1)

        return ufeat, ifeat

class GCMCLayer(nn.Module):
    def __init__(self,
                 rating_vals,
                 user_in_units,
                 movie_in_units,
                 msg_units,
                 out_units,
                 dropout_rate=0.0,
                 agg='stack',  # or 'sum'
                 agg_act=None,
                 out_act=None,
                 share_user_item_param=False,
                 device=None):
        super(GCMCLayer, self).__init__()
        self.rating_vals = rating_vals
        self.agg = agg
        self.share_user_item_param = share_user_item_param
        self.ufc = nn.Linear(msg_units, out_units)
        if share_user_item_param:
            self.ifc = self.ufc
        else:
            self.ifc = nn.Linear(msg_units, out_units)
        if agg == 'stack':
            # divide the original msg unit size by number of ratings to keep
            # the dimensionality
            assert msg_units % len(rating_vals) == 0
            msg_units = msg_units // len(rating_vals)
        self.dropout = nn.Dropout(dropout_rate)
        self.W_r = nn.ParameterDict()
        subConv = {}
        subConv2 = {}
        for rating in rating_vals:
            # PyTorch parameter name can't contain "."
            rating = to_etype_name(rating)
            rev_rating = 'rev-%s' % rating
            if share_user_item_param and user_in_units == movie_in_units:
                self.W_r[rating] = nn.Parameter(th.randn(user_in_units, msg_units))
                self.W_r['rev-%s' % rating] = self.W_r[rating]
                subConv[rating] = GCMCGraphGAT(user_in_units,
                                                msg_units,
                                                weight=False,
                                                device=device,
                                                dropout_rate=dropout_rate)
                subConv[rev_rating] = GCMCGraphGAT(user_in_units,
                                                    msg_units,
                                                    weight=False,
                                                    device=device,
                                                    dropout_rate=dropout_rate)
            else:
                self.W_r = None
                subConv[rating] = GCMCGraphConv(user_in_units,
                                                msg_units,
                                                weight=True,
                                                device=device,
                                                dropout_rate=dropout_rate)
                subConv[rev_rating] = GCMCGraphConv(movie_in_units,
                                                    msg_units,
                                                    weight=True,
                                                    device=device,
                                                    dropout_rate=dropout_rate)
                subConv2[rating] = GCMCGraphConv(msg_units,
                                                msg_units,
                                                weight=True,
                                                device=device,
                                                dropout_rate=dropout_rate)
                subConv2[rev_rating] = GCMCGraphConv(msg_units,
                                                    msg_units,
                                                    weight=True,
                                                    device=device,
                                                    dropout_rate=dropout_rate)
        self.conv = dglnn.HeteroGraphConv(subConv, aggregate=agg)
        # self.conv2 = dglnn.HeteroGraphConv(subConv2, aggregate=agg)
        self.agg_act = get_activation(agg_act)
        self.out_act = get_activation(out_act)
        self.device = device
        self.reset_parameters()

    def partial_to(self, device):
        """Put parameters into device except W_r

        Parameters
        ----------
        device : torch device
            Which device the parameters are put in.
        """
        assert device == self.device
        if device is not None:
            self.ufc.cuda(device)
            if self.share_user_item_param is False:
                self.ifc.cuda(device)
            self.dropout.cuda(device)

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, graph, ufeat=None, ifeat=None):
        in_feats = {'user' : ufeat, 'item' : ifeat}
        mod_args = {}
        for i, rating in enumerate(self.rating_vals):
            rating = to_etype_name(rating)
            rev_rating = 'rev-%s' % rating
            mod_args[rating] = (self.W_r[rating] if self.W_r is not None else None,)
            mod_args[rev_rating] = (self.W_r[rev_rating] if self.W_r is not None else None,)
        out_feats = self.conv(graph, in_feats, mod_args=mod_args)
        # out_feats['item'] = out_feats['item'].squeeze(1)
        # out_feats['user'] = out_feats['user'].squeeze(1)
        # out_feats = self.conv2(graph,out_feats, mod_args=mod_args)
        ufeat = out_feats['user']
        ifeat = out_feats['item']
        ufeat = ufeat.view(ufeat.shape[0], -1)
        ifeat = ifeat.view(ifeat.shape[0], -1)

        # fc and non-linear
        ufeat = self.agg_act(ufeat)
        ifeat = self.agg_act(ifeat)

        ufeat = self.dropout(ufeat)
        ifeat = self.dropout(ifeat)

        ufeat = self.ufc(ufeat)
        ifeat = self.ifc(ifeat)

        return self.out_act(ufeat), self.out_act(ifeat)

class BiDecoder(nn.Module):
    r"""Bi-linear decoder.

    Given a bipartite graph G, for each edge (i, j) ~ G, compute the likelihood
    of it being class r by:

    .. math::
        p(M_{ij}=r) = \text{softmax}(u_i^TQ_rv_j)

    The trainable parameter :math:`Q_r` is further decomposed to a linear
    combination of basis weight matrices :math:`P_s`:

    .. math::
        Q_r = \sum_{s=1}^{b} a_{rs}P_s

    Parameters
    ----------
    in_units : int
        Size of input user and movie features
    num_classes : int
        Number of classes.
    num_basis : int, optional
        Number of basis. (Default: 2)
    dropout_rate : float, optional
        Dropout raite (Default: 0.0)
    """
    def __init__(self,
                 in_units,
                 num_classes,
                 num_basis=1,
                 dropout_rate=0.0):
        super(BiDecoder, self).__init__()
        self._num_basis = num_basis
        self.dropout = nn.Dropout(dropout_rate)
        self.Ps = nn.ParameterList(
            nn.Parameter(th.randn(in_units, in_units))
            for _ in range(num_basis))
        self.combine_basis = nn.Linear(self._num_basis, num_classes, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, graph, ufeat, ifeat):
        """Forward function.

        Parameters
        ----------
        graph : DGLHeteroGraph
            "Flattened" user-movie graph with only one edge type.
        ufeat : th.Tensor
            User embeddings. Shape: (|V_u|, D)
        ifeat : th.Tensor
            Movie embeddings. Shape: (|V_m|, D)

        Returns
        -------
        th.Tensor
            Predicting scores for each user-movie edge.
        """
        with graph.local_scope():
            ufeat = self.dropout(ufeat)
            ifeat = self.dropout(ifeat)
            graph.nodes['item'].data['h'] = ifeat
            basis_out = []
            for i in range(self._num_basis):
                graph.nodes['user'].data['h'] = ufeat @ self.Ps[i]
                graph.apply_edges(fn.u_dot_v('h', 'h', 'sr'))
                basis_out.append(graph.edata['sr'])
            out = th.cat(basis_out, dim=1)
            out = self.combine_basis(out)
        return out

    def inference(self, uidfeat, ifeat):
        unrate_predict = (uidfeat @ self.Ps[0]) @ ifeat.T
        rate_predict = (uidfeat @ self.Ps[0]) @ ifeat.T
        prediction = th.exp(rate_predict) / (th.exp(rate_predict) + th.exp(unrate_predict))
        return prediction

class DenseBiDecoder(nn.Module):
    r"""Dense bi-linear decoder.

    Dense implementation of the bi-linear decoder used in GCMC. Suitable when
    the graph can be efficiently represented by a pair of arrays (one for source
    nodes; one for destination nodes).

    Parameters
    ----------
    in_units : int
        Size of input user and movie features
    num_classes : int
        Number of classes.
    num_basis : int, optional
        Number of basis. (Default: 2)
    dropout_rate : float, optional
        Dropout raite (Default: 0.0)
    """
    def __init__(self,
                 in_units,
                 num_classes,
                 num_basis=2,
                 dropout_rate=0.0):
        super().__init__()
        self._num_basis = num_basis
        self.dropout = nn.Dropout(dropout_rate)
        self.P = nn.Parameter(th.randn(num_basis, in_units, in_units))
        self.combine_basis = nn.Linear(self._num_basis, num_classes, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, ufeat, ifeat):
        """Forward function.

        Compute logits for each pair ``(ufeat[i], ifeat[i])``.

        Parameters
        ----------
        ufeat : th.Tensor
            User embeddings. Shape: (B, D)
        ifeat : th.Tensor
            Movie embeddings. Shape: (B, D)

        Returns
        -------
        th.Tensor
            Predicting scores for each user-movie edge. Shape: (B, num_classes)
        """
        ufeat = self.dropout(ufeat)
        ifeat = self.dropout(ifeat)
        out = th.einsum('ai,bij,aj->ab', ufeat, self.P, ifeat)
        out = self.combine_basis(out)
        return out

def dot_or_identity(A, B, device=None):
    # if A is None, treat as identity matrix
    if A is None:
        return B
    elif len(A.shape) == 1:
        if device is None:
            return B[A]
        else:
            return B[A].to(device)
    else:
        return A @ B
