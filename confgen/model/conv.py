import torch
from torch import nn, Tensor
from typing import Optional
import math
import torch.nn.functional as F
from ..molecule.features import get_atom_feature_dims, get_bond_feature_dims
from torch_geometric.utils import softmax
from torch_scatter import scatter


class MetaLayer(nn.Module):
    def __init__(
        self,
        edge_model,
        node_model,
        global_model,
        aggregate_edges_for_node_fn=None,
        aggregate_edges_for_globals_fn=None,
        aggregate_nodes_for_globals_fn=None,
        node_attn=False,
        emb_dim=None,
        global_attn=False,
    ):
        super().__init__()
        self.edge_model = edge_model
        self.node_model = node_model
        self.global_model = global_model
        self.aggregate_edges_for_node_fn = aggregate_edges_for_node_fn
        self.aggregate_edges_for_globals_fn = aggregate_edges_for_globals_fn
        self.aggregate_nodes_for_globals_fn = aggregate_nodes_for_globals_fn
        if node_attn:
            self.node_attn = NodeAttn(emb_dim, num_heads=None)
        else:
            self.node_attn = None
        if global_attn and self.global_model is not None:
            self.global_node_attn = GlobalAttn(emb_dim, num_heads=None)
            self.global_edge_attn = GlobalAttn(emb_dim, num_heads=None)
        else:
            self.global_node_attn = None
            self.global_edge_attn = None

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        u: Tensor,
        edge_batch: Tensor,
        node_batch: Tensor,
        num_nodes: None,
        num_edges: None,
    ):
        row = edge_index[0]
        col = edge_index[1]
        if self.edge_model is not None:
            row = edge_index[0]
            col = edge_index[1]
            sent_attributes = x[row]
            received_attributes = x[col]
            global_edges = torch.repeat_interleave(u, num_edges, dim=0)
            feat_list = [edge_attr, sent_attributes, received_attributes, global_edges]
            concat_feat = torch.cat(feat_list, dim=1)
            edge_attr = self.edge_model(concat_feat)

        if self.node_model is not None and self.node_attn is None:
            sent_attributes = self.aggregate_edges_for_node_fn(edge_attr, row, size=x.size(0))
            received_attributes = self.aggregate_edges_for_node_fn(edge_attr, col, size=x.size(0))
            global_nodes = torch.repeat_interleave(u, num_nodes, dim=0)
            feat_list = [x, sent_attributes, received_attributes, global_nodes]
            x = self.node_model(torch.cat(feat_list, dim=1))
        elif self.node_model is not None:
            max_node_id = x.size(0)
            sent_attributes = self.node_attn(x[row], x[col], edge_attr, row, max_node_id)
            received_attributes = self.node_attn(x[col], x[row], edge_attr, col, max_node_id)
            global_nodes = torch.repeat_interleave(u, num_nodes, dim=0)
            feat_list = [x, sent_attributes, received_attributes, global_nodes]
            x = self.node_model(torch.cat(feat_list, dim=1))

        if self.global_model is not None and self.global_node_attn is None:
            n_graph = u.size(0)
            node_attributes = self.aggregate_nodes_for_globals_fn(x, node_batch, size=n_graph)
            edge_attributes = self.aggregate_edges_for_globals_fn(
                edge_attr, edge_batch, size=n_graph
            )
            feat_list = [u, node_attributes, edge_attributes]
            u = self.global_model(torch.cat(feat_list, dim=-1))
        elif self.global_model is not None:
            n_graph = u.size(0)
            node_attributes = self.global_node_attn(
                torch.repeat_interleave(u, num_nodes, dim=0), x, node_batch, dim_size=n_graph
            )
            edge_attributes = self.global_edge_attn(
                torch.repeat_interleave(u, num_edges, dim=0),
                edge_attr,
                edge_batch,
                dim_size=n_graph,
            )
            feat_list = [u, node_attributes, edge_attributes]
            u = self.global_model(torch.cat(feat_list, dim=-1))

        return x, edge_attr, u


class DropoutIfTraining(nn.Module):
    """
    Borrow this implementation from deepmind
    """

    def __init__(self, p=0.0, submodule=None):
        super().__init__()
        assert p >= 0 and p <= 1
        self.p = p
        self.submodule = submodule if submodule else nn.Identity()

    def forward(self, x):
        x = self.submodule(x)
        newones = x.new_ones((x.size(0), 1))
        newones = F.dropout(newones, p=self.p, training=self.training)
        out = x * newones
        return out


class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        output_sizes,
        use_layer_norm=False,
        activation=nn.ReLU,
        dropout=0.0,
        layernorm_before=False,
        use_bn=False,
    ):
        super().__init__()
        module_list = []
        if not use_bn:
            if layernorm_before:
                module_list.append(nn.LayerNorm(input_size))
            for size in output_sizes:
                module_list.append(nn.Linear(input_size, size))
                if size != 1:
                    module_list.append(activation())
                    if dropout > 0:
                        module_list.append(nn.Dropout(dropout))
                input_size = size
            if not layernorm_before and use_layer_norm:
                module_list.append(nn.LayerNorm(input_size))
        else:
            for size in output_sizes:
                module_list.append(nn.Linear(input_size, size))
                if size != 1:
                    module_list.append(nn.BatchNorm1d(size))
                    module_list.append(activation())
                input_size = size

        self.module_list = nn.ModuleList(module_list)
        self.reset_parameters()

    def reset_parameters(self):
        for item in self.module_list:
            if hasattr(item, "reset_parameters"):
                item.reset_parameters()

    def forward(self, x):
        for item in self.module_list:
            x = item(x)
        return x


class MLPwoLastAct(nn.Module):
    def __init__(
        self,
        input_size,
        output_sizes,
        use_layer_norm=False,
        activation=nn.ReLU,
        dropout=0.0,
        layernorm_before=False,
        use_bn=False,
    ):
        super().__init__()
        module_list = []
        if not use_bn:
            if layernorm_before:
                module_list.append(nn.LayerNorm(input_size))

            if dropout > 0:
                module_list.append(nn.Dropout(dropout))
            for i, size in enumerate(output_sizes):
                module_list.append(nn.Linear(input_size, size))
                if i < len(output_sizes) - 1:
                    module_list.append(activation())
                input_size = size
            if not layernorm_before and use_layer_norm:
                module_list.append(nn.LayerNorm(input_size))
        else:
            for i, size in enumerate(output_sizes):
                module_list.append(nn.Linear(input_size, size))
                if i < len(output_sizes) - 1:
                    module_list.append(nn.BatchNorm1d(size))
                    module_list.append(activation())
                input_size = size

        self.module_list = nn.ModuleList(module_list)
        self.reset_parameters()

    def reset_parameters(self):
        for item in self.module_list:
            if hasattr(item, "reset_parameters"):
                item.reset_parameters()

    def forward(self, x):
        for item in self.module_list:
            x = item(x)
        return x


class AtomEncoder(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.m = nn.Linear(sum(get_atom_feature_dims()), emb_dim)

    def forward(self, x):
        return self.m(x)


class BondEncoder(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.m = nn.Linear(sum(get_bond_feature_dims()), emb_dim)

    def forward(self, x):
        return self.m(x)


class GatedLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_c):
        super().__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(dim_c, dim_out, bias=False)
        self._hyper_gate = nn.Linear(dim_c, dim_out)

    def forward(self, x, context):
        gate = torch.sigmoid(self._hyper_gate(context))
        bias = self._hyper_bias(context)
        return self._layer(x) * gate + bias


class NodeAttn(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        self.emb_dim = emb_dim
        if num_heads is None:
            num_heads = emb_dim // 64
        self.num_heads = num_heads
        assert self.emb_dim % self.num_heads == 0

        self.w1 = nn.Linear(3 * emb_dim, emb_dim)
        self.w2 = nn.Parameter(torch.zeros((self.num_heads, self.emb_dim // self.num_heads)))
        self.w3 = nn.Linear(2 * emb_dim, emb_dim)
        self.head_dim = self.emb_dim // self.num_heads
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w1.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w2, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w3.weight, gain=1 / math.sqrt(2))

    def forward(self, q, k_v, k_e, index, nnode):
        """
        q: [N, C]
        k: [N, 2*c]
        v: [N, 2*c]
        """
        x = torch.cat([q, k_v, k_e], dim=1)
        x = self.w1(x).view(-1, self.num_heads, self.head_dim)
        x = F.leaky_relu(x)
        attn_weight = torch.einsum("nhc,hc->nh", x, self.w2).unsqueeze(-1)
        attn_weight = softmax(attn_weight, index)

        v = torch.cat([k_v, k_e], dim=1)
        v = self.w3(v).view(-1, self.num_heads, self.head_dim)
        x = (attn_weight * v).reshape(-1, self.emb_dim)
        x = scatter(x, index, dim=0, reduce="sum", dim_size=nnode)
        return x


class GlobalAttn(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        self.emb_dim = emb_dim
        if num_heads is None:
            num_heads = emb_dim // 64
        self.num_heads = num_heads
        assert self.emb_dim % self.num_heads == 0
        self.w1 = nn.Linear(2 * emb_dim, emb_dim)
        self.w2 = nn.Parameter(torch.zeros(self.num_heads, self.emb_dim // self.num_heads))
        self.head_dim = self.emb_dim // self.num_heads
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.w1.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.w2, gain=1 / math.sqrt(2))

    def forward(self, q, k, index, dim_size):
        x = torch.cat([q, k], dim=1)
        x = self.w1(x).view(-1, self.num_heads, self.head_dim)
        x = F.leaky_relu(x)
        attn_weight = torch.einsum("nhc,hc->nh", x, self.w2).unsqueeze(-1)
        attn_weight = softmax(attn_weight, index)

        v = k.view(-1, self.num_heads, self.head_dim)
        x = (attn_weight * v).reshape(-1, self.emb_dim)
        x = scatter(x, index, dim=0, reduce="sum", dim_size=dim_size)
        return x

