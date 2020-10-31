import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, degree

from logger import config_logger
from utils import uniform

logger = config_logger('Model')


class GRUEncoder(nn.Module):
    def __init__(self, embed_matrix, args):
        super(GRUEncoder, self).__init__()
        self.args = args
        self.bi = self.args.bidirectional
        self.hidden_size = self.args.nfeats // 2 if self.bi else self.args.nfeats
        self.num_layers = self.args.num_layers
        self.embed_matrix = embed_matrix
        self.pad_id = self.args.pad_id
        self.relPoolType = self.args.relPoolType
        self.dropout = self.args.dropout

        self.embed = nn.Embedding(num_embeddings=self.embed_matrix.shape[0],
                                  embedding_dim=self.embed_matrix.shape[1],
                                  padding_idx=0)

        self.encoder = nn.GRU(self.embed_matrix.shape[1], self.hidden_size,
                              self.num_layers, dropout=self.dropout, batch_first=True, bidirectional=self.bi)

        self.embed.weight.data.copy_(torch.from_numpy(self.embed_matrix))

    def _encode(self, batch, doc_len):
        size, sort = torch.sort(doc_len, dim=0, descending=True)
        _, unsort = torch.sort(sort, dim=0)
        batch = torch.index_select(batch, dim=0, index=sort)
        embedded = self.embed(batch)
        packed = pack(embedded, size.data.tolist(), batch_first=True)
        encoded, h = self.encoder(packed)
        unpacked, _ = unpack(encoded, batch_first=True)
        unpacked = torch.index_select(unpacked, dim=0, index=unsort)
        h = torch.index_select(h, dim=1, index=unsort)
        return unpacked, h

    def _pool(self, unpacked, h):
        batchSize, Seqlength, temp = unpacked.size()

        if self.relPoolType == 'last':
            idx = 2 if self.bi else 1
            pooled = h[-idx:].transpose(0, 1).contiguous().view(batchSize, -1)
        elif self.relPoolType == 'mean':
            pooled = torch.mean(unpacked, dim=1)
        elif self.relPoolType == 'max':
            pooled, _ = torch.max(unpacked, dim=1)
        else:
            logger.error("The mode does not support relPoolType %s", self.relPoolType)
            exit(0)

        return pooled

    def _getFeatures(self, batch, doc_len):
        encoded, hidden = self._encode(batch, doc_len)
        hidden = self._pool(encoded, hidden)
        return hidden

    def forward(self, batch, doc_len):
        phrase_encode = self._getFeatures(batch, doc_len)
        return phrase_encode


class LinearEncoder(nn.Module):
    def __init__(self, args):
        super(LinearEncoder, self).__init__()
        self.args = args
        self.entPoolType = self.args.entPoolType
        self.lin = torch.nn.Linear(args.input_dim, args.nfeats, bias=True)

    def forward(self, x, edge):
        x = F.tanh(self.lin(x))
        if self.entPoolType == 'mean':
            x = torch.mean(x, dim=1)
        elif self.entPoolType == 'max':
            x, _ = torch.max(x, dim=1)
        else:
            logger.error("The mode does not support entPoolType %s", self.entPoolType)
            exit(0)

        return x


class GCNCov1(MessagePassing):
    def __init__(self, in_channels, out_channels, entPoolType):
        super(GCNCov1, self).__init__(aggr='add')
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.entPoolType = entPoolType

    def forward(self, x, edge_index):
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)

        if self.entPoolType == 'mean':
            x = torch.mean(x, dim=1)
        elif self.entPoolType == 'max':
            x, _ = torch.max(x, dim=1)
        else:
            logger.error("The mode does not support entPoolType %s", self.entPoolType)
            exit(0)

        return self.propagate(edge_index, Shape=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, Shape):
        row, col = edge_index

        deg = degree(row, Shape[0], dtype=x_j.dtype)

        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return norm.view(-1, 1) * x_j

    def update(self, h_prime, x):
        return h_prime


class GCNCov2(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNCov2, self).__init__(aggr='add')
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)

        return self.propagate(edge_index, Shape=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, Shape):
        row, col = edge_index

        deg = degree(row, Shape[0], dtype=x_j.dtype)

        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return norm.view(-1, 1) * x_j

    def update(self, h_prime, x):
        return h_prime


class GcnNet(nn.Module):
    def __init__(self, in_channels, out_channels, entPoolType):
        super(GcnNet, self).__init__()
        self.gcn1 = GCNCov1(in_channels, 300, entPoolType)
        self.gcn2 = GCNCov2(300, out_channels)

    def forward(self, x, edge_index):
        h = F.leaky_relu(self.gcn1(x, edge_index))
        np_embedding = self.gcn2(h, edge_index)
        return np_embedding


class LAN(MessagePassing):
    def __init__(self, in_channels, out_channels, entPoolType):
        super(LAN, self).__init__(aggr='add')
        self.entPoolType = entPoolType
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        x = self.lin(x)
        if self.entPoolType == 'mean':
            x = torch.mean(x, dim=1)
        elif self.entPoolType == 'max':
            x, _ = torch.max(x, dim=1)
        else:
            logger.error("The mode does not support entPoolType %s", self.entPoolType)
            exit(0)

        return self.propagate(edge_index, Shape=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, Shape):
        row, col = edge_index

        deg = degree(row, Shape[0], dtype=x_j.dtype)

        deg_inv = deg.pow(-1.0)
        return deg_inv[col].view(-1, 1) * x_j

    def update(self, h_prime, x):
        return x / 2 + h_prime / 2


class RGCNConv(MessagePassing):
    r"""The relational graph convolutional operator from the `"Modeling
    Relational Data with Graph Convolutional Networks"
    <https://arxiv.org/abs/1703.06103>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}_{\textrm{root}} \cdot
        \mathbf{x}_i + \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_r(i)}
        \frac{1}{|\mathcal{N}_r(i)|} \mathbf{\Theta}_r \cdot \mathbf{x}_j,

    where :math:`\mathcal{R}` denotes the set of relations, *i.e.* edge types.
    Edge type needs to be a one-dimensional :obj:`torch.long` tensor which
    stores a relation identifier
    :math:`\in \{ 0, \ldots, |\mathcal{R}| - 1\}` for each edge.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        num_relations (int): Number of relations.
        num_bases (int): Number of bases used for basis-decomposition.
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels, num_relations, num_bases,
                 root_weight=True, bias=True, **kwargs):
        super(RGCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases

        self.basis = nn.Parameter(torch.Tensor(num_bases, in_channels, out_channels))
        self.att = nn.Parameter(torch.Tensor(num_relations, num_bases))

        if root_weight:
            self.root = nn.Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.num_bases * self.in_channels
        uniform(size, self.basis)
        uniform(size, self.att)
        uniform(size, self.root)
        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_type, edge_norm=None, size=None):
        """"""
        return self.propagate(edge_index, size=size, x=x, edge_type=edge_type,
                              edge_norm=edge_norm)

    def message(self, x_j, edge_index_j, edge_type, edge_norm):
        w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))

        # If no node features are given, we implement a simple embedding
        # loopkup based on the target node index and its edge type.
        if x_j is None:
            w = w.view(-1, self.out_channels)
            index = edge_type * self.in_channels + edge_index_j
            out = torch.index_select(w, 0, index)
        else:
            w = w.view(self.num_relations, self.in_channels, self.out_channels)
            w = torch.index_select(w, 0, edge_type)
            out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out, x):
        if self.root is not None:
            if x is None:
                out = aggr_out + self.root
            else:
                out = aggr_out + torch.matmul(x, self.root)

        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_relations)


class RGCNNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, entPoolType, num_relations, num_bases):
        super(RGCNNet, self).__init__()
        self.entPoolType = entPoolType
        self.conv1 = RGCNConv(
            in_channels, out_channels, num_relations, num_bases=num_bases)
        self.lin = torch.nn.Linear(in_channels, 300)

    def forward(self, x, edge_index, edge_type, edge_norm):
        x = self.lin(x)

        if self.entPoolType == 'mean':
            x = torch.mean(x, dim=1)
        elif self.entPoolType == 'max':
            x, _ = torch.max(x, dim=1)
        else:
            logger.error("The mode does not support entPoolType %s", self.entPoolType)
            exit(0)

        x = self.conv1(x, edge_index, edge_type, edge_norm)

        return x
