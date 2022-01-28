import torch
from torch import nn
from torch_geometric.nn import (
    global_add_pool,
    global_mean_pool,
    global_max_pool,
)
from ..molecule.features import get_atom_feature_dims, get_bond_feature_dims
from .conv import DropoutIfTraining, MLP, MLPwoLastAct, MetaLayer, GatedLinear
import torch.nn.functional as F
import numpy as np
from confgen.utils.utils import get_random_rotation_3d, clip_norm

_REDUCER_NAMES = {
    "sum": global_add_pool,
    "mean": global_mean_pool,
    "max": global_max_pool,
}


class GNN(nn.Module):
    def __init__(
        self,
        mlp_hidden_size: int = 512,
        mlp_layers: int = 2,
        latent_size: int = 256,
        use_layer_norm: bool = False,
        num_message_passing_steps: int = 12,
        global_reducer: str = "sum",
        node_reducer: str = "sum",
        dropedge_rate: float = 0.1,
        dropnode_rate: float = 0.1,
        dropout: float = 0.1,
        graph_pooling: str = "sum",
        layernorm_before: bool = False,
        pooler_dropout: float = 0.0,
        encoder_dropout: float = 0.0,
        use_bn: bool = False,
        vae_beta: float = 1.0,
        decoder_layers: int = None,
        reuse_prior: bool = False,
        cycle: int = 1,
        pred_pos_residual: bool = False,
        node_attn: bool = False,
        shared_decoder: bool = False,
        shared_output: bool = False,
        global_attn: bool = False,
        sample_beta: float = 1,
        use_global: bool = False,
        sg_pos: bool = False,
        use_ss: bool = False,
        rand_aug: bool = False,
        no_3drot: bool = False,
        not_origin: bool = False,
    ):
        super().__init__()

        self.encoder_edge = MLP(
            sum(get_bond_feature_dims()),
            [mlp_hidden_size] * mlp_layers + [latent_size],
            use_layer_norm=use_layer_norm,
        )
        self.encoder_node = MLP(
            sum(get_atom_feature_dims()),
            [mlp_hidden_size] * mlp_layers + [latent_size],
            use_layer_norm=use_layer_norm,
        )
        self.global_init = nn.Parameter(torch.zeros((1, latent_size)))

        # prior conf
        self.prior_conf_gnns = nn.ModuleList()
        self.prior_conf_pos = nn.ModuleList()
        for _ in range(num_message_passing_steps):
            edge_model = DropoutIfTraining(
                p=dropedge_rate,
                submodule=MLP(
                    latent_size * 4,
                    [mlp_hidden_size] * mlp_layers + [latent_size],
                    use_layer_norm=use_layer_norm,
                    layernorm_before=layernorm_before,
                    dropout=encoder_dropout,
                    use_bn=use_bn,
                ),
            )
            node_model = DropoutIfTraining(
                p=dropnode_rate,
                submodule=MLP(
                    latent_size * 4,
                    [mlp_hidden_size] * mlp_layers + [latent_size],
                    use_layer_norm=use_layer_norm,
                    layernorm_before=layernorm_before,
                    dropout=encoder_dropout,
                    use_bn=use_bn,
                ),
            )
            global_model = MLP(
                latent_size * 3,
                [mlp_hidden_size] * mlp_layers + [latent_size],
                use_layer_norm=use_layer_norm,
                layernorm_before=layernorm_before,
                dropout=encoder_dropout,
                use_bn=use_bn,
            )
            self.prior_conf_gnns.append(
                MetaLayer(
                    edge_model=edge_model,
                    node_model=node_model,
                    global_model=global_model,
                    aggregate_edges_for_node_fn=_REDUCER_NAMES[node_reducer],
                    aggregate_edges_for_globals_fn=_REDUCER_NAMES[global_reducer],
                    aggregate_nodes_for_globals_fn=_REDUCER_NAMES[global_reducer],
                    node_attn=node_attn,
                    emb_dim=latent_size,
                    global_attn=global_attn,
                )
            )
            self.prior_conf_pos.append(MLPwoLastAct(latent_size, [latent_size, 3]))

        # encoder
        self.encoder_gnns = nn.ModuleList()
        for i in range(num_message_passing_steps):
            edge_model = DropoutIfTraining(
                p=dropedge_rate,
                submodule=MLP(
                    latent_size * 4,
                    [mlp_hidden_size] * mlp_layers + [latent_size],
                    use_layer_norm=use_layer_norm,
                    layernorm_before=layernorm_before,
                    dropout=encoder_dropout,
                    use_bn=use_bn,
                ),
            )
            node_model = DropoutIfTraining(
                p=dropnode_rate,
                submodule=MLP(
                    latent_size * 4,
                    [mlp_hidden_size] * mlp_layers + [latent_size],
                    use_layer_norm=use_layer_norm,
                    layernorm_before=layernorm_before,
                    dropout=encoder_dropout,
                    use_bn=use_bn,
                ),
            )
            if i == num_message_passing_steps - 1 and not use_global:
                global_model = None
            else:
                global_model = MLP(
                    latent_size * 3,
                    [mlp_hidden_size] * mlp_layers + [latent_size],
                    use_layer_norm=use_layer_norm,
                    layernorm_before=layernorm_before,
                    dropout=encoder_dropout,
                    use_bn=use_bn,
                )
            self.encoder_gnns.append(
                MetaLayer(
                    edge_model=edge_model,
                    node_model=node_model,
                    global_model=global_model,
                    aggregate_edges_for_node_fn=_REDUCER_NAMES[node_reducer],
                    aggregate_edges_for_globals_fn=_REDUCER_NAMES[global_reducer],
                    aggregate_nodes_for_globals_fn=_REDUCER_NAMES[global_reducer],
                    node_attn=node_attn,
                    emb_dim=latent_size,
                    global_attn=global_attn,
                )
            )
        self.encoder_head = MLPwoLastAct(latent_size, [latent_size, 2 * latent_size], use_bn=use_bn)

        # decoder
        self.decoder_gnns = nn.ModuleList()
        self.decoder_pos = nn.ModuleList()
        decoder_layers = num_message_passing_steps if decoder_layers is None else decoder_layers
        for i in range(decoder_layers):
            if (not shared_decoder) or i == 0:
                edge_model = DropoutIfTraining(
                    p=dropedge_rate,
                    submodule=MLP(
                        latent_size * 4,
                        [mlp_hidden_size] * mlp_layers + [latent_size],
                        use_layer_norm=use_layer_norm,
                        layernorm_before=layernorm_before,
                        dropout=encoder_dropout,
                        use_bn=use_bn,
                    ),
                )
                node_model = DropoutIfTraining(
                    p=dropnode_rate,
                    submodule=MLP(
                        latent_size * 4,
                        [mlp_hidden_size] * mlp_layers + [latent_size],
                        use_layer_norm=use_layer_norm,
                        layernorm_before=layernorm_before,
                        dropout=encoder_dropout,
                        use_bn=use_bn,
                    ),
                )
                if i == decoder_layers - 1:
                    global_model = None
                else:
                    global_model = MLP(
                        latent_size * 3,
                        [mlp_hidden_size] * mlp_layers + [latent_size],
                        use_layer_norm=use_layer_norm,
                        layernorm_before=layernorm_before,
                        dropout=encoder_dropout,
                        use_bn=use_bn,
                    )
                self.decoder_gnns.append(
                    MetaLayer(
                        edge_model=edge_model,
                        node_model=node_model,
                        global_model=global_model,
                        aggregate_edges_for_node_fn=_REDUCER_NAMES[node_reducer],
                        aggregate_edges_for_globals_fn=_REDUCER_NAMES[global_reducer],
                        aggregate_nodes_for_globals_fn=_REDUCER_NAMES[global_reducer],
                        node_attn=node_attn,
                        emb_dim=latent_size,
                        global_attn=global_attn,
                    )
                )
                if (not shared_output) or i == 0:
                    self.decoder_pos.append(MLPwoLastAct(latent_size, [latent_size, 3]))
                else:
                    self.decoder_pos.append(self.decoder_pos[-1])
            else:
                self.decoder_gnns.append(self.decoder_gnns[-1])
                self.decoder_pos.append(self.decoder_pos[-1])

        # self.dist_pred_layer = MLPwoLastAct(latent_size, [latent_size, 1])
        self.pooling = _REDUCER_NAMES[graph_pooling]

        self.pos_embedding = MLP(3, [latent_size, latent_size])
        self.dis_embedding = MLP(1, [latent_size, latent_size])

        self.latent_size = latent_size
        self.dropout = dropout
        self.vae_beta = vae_beta
        self.reuse_prior = reuse_prior
        self.cycle = cycle
        self.pred_pos_residual = pred_pos_residual
        self.sample_beta = sample_beta
        self.use_global = use_global
        self.sg_pos = sg_pos
        self.not_origin = not_origin

        # self supervised
        self.use_ss = use_ss
        if self.use_ss:
            self.projection_head = MLPwoLastAct(
                latent_size, [mlp_hidden_size, latent_size], use_bn=True
            )
            self.prediction_head = MLPwoLastAct(
                latent_size, [mlp_hidden_size, latent_size], use_bn=True
            )

        # random augmentation
        self.rand_aug = rand_aug
        self.no_3drot = no_3drot

    def forward(self, batch, sample=False):
        (x, edge_index, edge_attr, node_batch, num_nodes, num_edges, num_graphs,) = (
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
            batch.n_nodes,
            batch.n_edges,
            batch.num_graphs,
        )

        onehot_x = one_hot_atoms(x)
        onehot_edge_attr = one_hot_bonds(edge_attr)

        graph_idx = torch.arange(num_graphs).to(x.device)
        edge_batch = torch.repeat_interleave(graph_idx, num_edges, dim=0)

        x_embed = self.encoder_node(onehot_x)
        edge_attr_embed = self.encoder_edge(onehot_edge_attr)
        u_embed = self.global_init.expand(num_graphs, -1)

        extra_output = {}

        # prior conf

        cur_pos = x_embed.new_zeros((x_embed.size(0), 3)).uniform_(-1, 1)
        pos_list = []

        x = x_embed
        edge_attr = edge_attr_embed
        u = u_embed
        for i, layer in enumerate(self.prior_conf_gnns):
            extended_x, extended_edge_attr = self.extend_x_edge(cur_pos, x, edge_attr, edge_index)
            x_1, edge_attr_1, u_1 = layer(
                extended_x,
                edge_index,
                extended_edge_attr,
                u,
                edge_batch,
                node_batch,
                num_nodes,
                num_edges,
            )
            x = F.dropout(x_1, p=self.dropout, training=self.training) + x
            edge_attr = F.dropout(edge_attr_1, p=self.dropout, training=self.training) + edge_attr
            u = F.dropout(u_1, p=self.dropout, training=self.training) + u
            if self.pred_pos_residual:
                delta_pos = self.prior_conf_pos[i](x)
                cur_pos = self.move2origin(cur_pos + delta_pos, batch)
            else:
                cur_pos = self.prior_conf_pos[i](x)
                cur_pos = self.move2origin(cur_pos, batch)
            cur_pos = self.random_augmentation(cur_pos, batch)
            pos_list.append(cur_pos)
        # extra_output["prior_last_edge"] = self.dist_pred_layer(edge_attr)
        extra_output["prior_pos_list"] = pos_list

        prior_output = [x, edge_attr, u]

        if not sample:
            # encoder
            x = x_embed
            edge_attr = edge_attr_embed
            u = u_embed
            cur_pos = self.move2origin(batch.pos, batch)
            if not self.no_3drot:
                cur_pos = get_random_rotation_3d(cur_pos)
            for i, layer in enumerate(self.encoder_gnns):
                extended_x, extended_edge_attr = self.extend_x_edge(
                    cur_pos, x, edge_attr, edge_index
                )
                x_1, edge_attr_1, u_1 = layer(
                    extended_x,
                    edge_index,
                    extended_edge_attr,
                    u,
                    edge_batch,
                    node_batch,
                    num_nodes,
                    num_edges,
                )
                x = F.dropout(x_1, p=self.dropout, training=self.training) + x
                edge_attr = (
                    F.dropout(edge_attr_1, p=self.dropout, training=self.training) + edge_attr
                )
                u = F.dropout(u_1, p=self.dropout, training=self.training) + u
                if self.use_ss:
                    cur_pos = get_random_rotation_3d(cur_pos)

            if self.use_global:
                aggregated_feat = u
            else:
                aggregated_feat = self.pooling(x, node_batch)

            if self.use_ss:
                extra_output["query_feat"] = self.prediction_head(
                    self.projection_head(aggregated_feat)
                )
                with torch.no_grad():
                    x = x_embed
                    edge_attr = edge_attr_embed
                    u = u_embed
                    for i, layer in enumerate(self.encoder_gnns):
                        extended_x, extended_edge_attr = self.extend_x_edge(
                            cur_pos, x, edge_attr, edge_index
                        )
                        x_1, edge_attr_1, u_1 = layer(
                            extended_x,
                            edge_index,
                            extended_edge_attr,
                            u,
                            edge_batch,
                            node_batch,
                            num_nodes,
                            num_edges,
                        )
                        x = F.dropout(x_1, p=self.dropout, training=self.training) + x
                        edge_attr = (
                            F.dropout(edge_attr_1, p=self.dropout, training=self.training)
                            + edge_attr
                        )
                        u = F.dropout(u_1, p=self.dropout, training=self.training) + u
                        cur_pos = get_random_rotation_3d(cur_pos)
                    if self.use_global:
                        aggregated_feat_1 = u
                    else:
                        aggregated_feat_1 = self.pooling(x, node_batch)

                    extra_output["key_feat"] = self.projection_head(aggregated_feat_1)

            latent = self.encoder_head(aggregated_feat)
            latent_mean, latent_logstd = torch.chunk(latent, chunks=2, dim=-1)
            extra_output["latent_mean"] = latent_mean
            extra_output["latent_logstd"] = latent_logstd

            # decoder
            z = self.reparameterization(latent_mean, latent_logstd)
        else:
            z = torch.randn_like(u_embed) * self.sample_beta
        z = torch.repeat_interleave(z, num_nodes, dim=0)
        if self.reuse_prior:
            x, edge_attr, u = prior_output
        else:
            x, edge_attr, u = x_embed, edge_attr_embed, u_embed

        cur_pos = pos_list[-1]
        pos_list = []

        for i, layer in enumerate(self.decoder_gnns):
            if i == len(self.decoder_gnns) - 1:
                cycle = self.cycle
            else:
                cycle = 1
            for _ in range(cycle):
                extended_x, extended_edge_attr = self.extend_x_edge(
                    cur_pos, x + z, edge_attr, edge_index
                )
                x_1, edge_attr_1, u_1 = layer(
                    extended_x,
                    edge_index,
                    extended_edge_attr,
                    u,
                    edge_batch,
                    node_batch,
                    num_nodes,
                    num_edges,
                )
                x = F.dropout(x_1, p=self.dropout, training=self.training) + x
                edge_attr = (
                    F.dropout(edge_attr_1, p=self.dropout, training=self.training) + edge_attr
                )
                u = F.dropout(u_1, p=self.dropout, training=self.training) + u
                if self.pred_pos_residual:
                    delta_pos = self.decoder_pos[i](x)
                    cur_pos = self.move2origin(cur_pos + delta_pos, batch)
                else:
                    cur_pos = self.decoder_pos[i](x)
                    cur_pos = self.move2origin(cur_pos, batch)
                cur_pos = self.random_augmentation(cur_pos, batch)
                pos_list.append(cur_pos)
                if self.sg_pos:
                    cur_pos = cur_pos.detach()

        # extra_output["decoder_last_edge"] = self.dist_pred_layer(edge_attr)
        return pos_list, extra_output

    def random_augmentation(self, pos, batch):
        if self.rand_aug and self.training:
            return get_random_rotation_3d(pos)
        else:
            return pos

    def reparameterization(self, mean, log_std):
        std = torch.exp(log_std)
        epsilon = torch.randn_like(std)
        z = mean + std * epsilon
        return z

    def move2origin(self, pos, batch):
        if self.not_origin:
            return pos
        pos_mean = global_mean_pool(pos, batch.batch)
        return pos - torch.repeat_interleave(pos_mean, batch.n_nodes, dim=0)

    def extend_x_edge(self, pos, x, edge_attr, edge_index):
        # extended_x = torch.cat([x, pos], dim=1)
        extended_x = x + self.pos_embedding(pos)
        row = edge_index[0]
        col = edge_index[1]
        sent_pos = pos[row]
        received_pos = pos[col]
        length = (sent_pos - received_pos).norm(dim=-1).unsqueeze(-1)
        # extended_edge_attr = torch.cat([edge_attr, length], dim=-1)
        extended_edge_attr = edge_attr + self.dis_embedding(length)
        return extended_x, extended_edge_attr

    def compute_loss(self, pos_list, extra_output, batch, args):
        loss_dict = {}
        loss = 0

        pos = batch.pos

        new_idx = GNN.update_iso(pos, pos_list[-1], batch)

        loss_tmp, _ = self.alignment_loss(
            pos, extra_output["prior_pos_list"][-1].index_select(0, new_idx), batch
        )
        loss = loss + loss_tmp
        loss_dict["loss_prior_pos"] = loss_tmp

        mean = extra_output["latent_mean"]
        log_std = extra_output["latent_logstd"]
        kld = -0.5 * torch.sum(1 + 2 * log_std - mean.pow(2) - torch.exp(2 * log_std), dim=-1)
        kld = kld.mean()
        loss = loss + kld * args.vae_beta
        loss_dict["loss_kld"] = kld

        loss_tmp, _ = self.alignment_loss(
            pos, pos_list[-1].index_select(0, new_idx), batch, clamp=args.clamp_dist
        )
        loss = loss + loss_tmp
        loss_dict["loss_pos_last"] = loss_tmp

        if args.aux_loss > 0:
            for i in range(len(pos_list) - 1):
                loss_tmp, _ = self.alignment_loss(
                    pos, pos_list[i].index_select(0, new_idx), batch, clamp=args.clamp_dist
                )
                loss = loss + loss_tmp * (args.aux_loss if i < len(pos_list) - args.cycle else 1.0)
                loss_dict[f"loss_pos_{i}"] = loss_tmp

        if args.ang_lam > 0 or args.bond_lam > 0:
            bond_loss, angle_loss = self.aux_loss(pos, pos_list[-1].index_select(0, new_idx), batch)
            loss_dict["bond_loss"] = bond_loss
            loss_dict["angle_loss"] = angle_loss
            loss = loss + args.bond_lam * bond_loss + args.ang_lam * angle_loss

        if self.use_ss:
            anchor = extra_output["query_feat"]
            positive = extra_output["key_feat"]
            anchor = anchor / torch.norm(anchor, dim=-1, keepdim=True)
            positive = positive / torch.norm(positive, dim=-1, keepdim=True)

            loss_tmp = torch.einsum("nc,nc->n", [anchor, positive]).mean()
            loss = loss - loss_tmp
            loss_dict[f"loss_ss"] = loss_tmp

        loss_dict["loss"] = loss
        return loss, loss_dict

    @staticmethod
    def aux_loss(pos_y, pos_x, batch):
        edge_index = batch.edge_index
        src = edge_index[0]
        tgt = edge_index[1]
        true_bond = torch.norm(pos_y[src] - pos_y[tgt], dim=-1)
        pred_bond = torch.norm(pos_x[src] - pos_x[tgt], dim=-1)
        bond_loss = torch.mean(F.l1_loss(pred_bond, true_bond))

        nei_src_index = batch.nei_src_index.view(-1)
        nei_tgt_index = batch.nei_tgt_index
        nei_tgt_mask = batch.nei_tgt_mask

        random_tgt_index = pos_y.new_zeros(nei_tgt_index.size()).uniform_()
        random_tgt_index = torch.where(
            nei_tgt_mask, pos_y.new_zeros(nei_tgt_index.size()), random_tgt_index
        )
        random_tgt_index_sort = torch.sort(random_tgt_index, descending=True, dim=0)[1][:2]
        tgt_1, tgt_2 = random_tgt_index_sort[0].unsqueeze(0), random_tgt_index_sort[1].unsqueeze(0)

        tgt_1 = torch.gather(nei_tgt_index, 0, tgt_1).view(-1)
        tgt_2 = torch.gather(nei_tgt_index, 0, tgt_2).view(-1)

        def get_angle(vec1, vec2):
            vec1 = vec1 / (torch.norm(vec1, keepdim=True, dim=-1) + 1e-6)
            vec2 = vec2 / (torch.norm(vec2, keepdim=True, dim=-1) + 1e-6)
            return torch.einsum("nc,nc->n", vec1, vec2)

        true_angle = get_angle(
            pos_y[tgt_1] - pos_y[nei_src_index], pos_y[tgt_2] - pos_y[nei_src_index]
        )
        pred_angle = get_angle(
            pos_x[tgt_1] - pos_x[nei_src_index], pos_x[tgt_2] - pos_x[nei_src_index]
        )
        angle_loss = torch.mean(F.l1_loss(pred_angle, true_angle))

        return bond_loss, angle_loss

    @staticmethod
    def quaternion_to_rotation_matrix(quaternion):
        q0 = quaternion[:, 0]
        q1 = quaternion[:, 1]
        q2 = quaternion[:, 2]
        q3 = quaternion[:, 3]

        r00 = 2 * (q0 * q0 + q1 * q1) - 1
        r01 = 2 * (q1 * q2 - q0 * q3)
        r02 = 2 * (q1 * q3 + q0 * q2)
        r10 = 2 * (q1 * q2 + q0 * q3)
        r11 = 2 * (q0 * q0 + q2 * q2) - 1
        r12 = 2 * (q2 * q3 - q0 * q1)
        r20 = 2 * (q1 * q3 - q0 * q2)
        r21 = 2 * (q2 * q3 + q0 * q1)
        r22 = 2 * (q0 * q0 + q3 * q3) - 1
        return torch.stack([r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=-1).reshape(-1, 3, 3)

    @staticmethod
    def alignment_loss(pos_y, pos_x, batch, clamp=None):
        with torch.no_grad():
            num_nodes = batch.n_nodes
            total_nodes = pos_y.shape[0]
            num_graphs = batch.num_graphs
            pos_y_mean = global_mean_pool(pos_y, batch.batch)
            pos_x_mean = global_mean_pool(pos_x, batch.batch)
            y = pos_y - torch.repeat_interleave(pos_y_mean, num_nodes, dim=0)
            x = pos_x - torch.repeat_interleave(pos_x_mean, num_nodes, dim=0)
            a = y + x
            b = y - x
            a = a.view(total_nodes, 1, 3)
            b = b.view(total_nodes, 3, 1)
            tmp0 = torch.cat(
                [b.new_zeros((1, 1, 1)).expand(total_nodes, -1, -1), -b.permute(0, 2, 1)], dim=-1
            )
            eye = torch.eye(3).to(a).unsqueeze(0).expand(total_nodes, -1, -1)
            a = a.expand(-1, 3, -1)
            tmp1 = torch.cross(eye, a, dim=-1)
            tmp1 = torch.cat([b, tmp1], dim=-1)
            tmp = torch.cat([tmp0, tmp1], dim=1)
            tmpb = torch.bmm(tmp.permute(0, 2, 1), tmp).view(total_nodes, -1)
            tmpb = global_mean_pool(tmpb, batch.batch).view(num_graphs, 4, 4)
            w, v = torch.linalg.eigh(tmpb)
            min_rmsd = w[:, 0]
            min_q = v[:, :, 0]
            rotation = GNN.quaternion_to_rotation_matrix(min_q)
            t = pos_y_mean - torch.einsum("kj,kij->ki", pos_x_mean, rotation)
            rotation = torch.repeat_interleave(rotation, num_nodes, dim=0)
            t = torch.repeat_interleave(t, num_nodes, dim=0)
        pos_x = torch.einsum("kj,kij->ki", pos_x, rotation) + t
        if clamp is None:
            loss = global_mean_pool((pos_y - pos_x).norm(dim=-1, keepdim=True), batch.batch).mean()
        else:
            loss = global_mean_pool((pos_y - pos_x).norm(dim=-1, keepdim=True), batch.batch)
            loss = torch.clamp(loss, min=clamp).mean()
        return loss, min_rmsd.mean()

    @staticmethod
    def alignment_loss_iso_onegraph(pos_y, pos_x, pos_y_mean, pos_x_mean, num_nodes, total_iso):
        with torch.no_grad():
            total_nodes = pos_y.shape[0]
            y = pos_y - pos_y_mean
            x = pos_x - pos_x_mean
            a = y + x
            b = y - x
            a = a.view(-1, 1, 3)
            b = b.view(-1, 3, 1)
            tmp0 = torch.cat(
                [b.new_zeros((1, 1, 1)).expand(total_nodes, -1, -1), -b.permute(0, 2, 1)], dim=-1
            )
            eye = torch.eye(3).to(a).unsqueeze(0).expand(total_nodes, -1, -1)
            a = a.expand(-1, 3, -1)
            tmp1 = torch.cross(eye, a, dim=-1)
            tmp1 = torch.cat([b, tmp1], dim=-1)
            tmp = torch.cat([tmp0, tmp1], dim=1)
            tmpb = torch.bmm(tmp.permute(0, 2, 1), tmp).view(-1, num_nodes, 16)
            tmpb = torch.mean(tmpb, dim=1).view(-1, 4, 4)
            w, v = torch.linalg.eigh(tmpb)
            min_q = v[:, :, 0]
            rotation = GNN.quaternion_to_rotation_matrix(min_q)
            t = pos_y_mean - torch.einsum("kj,kij->ki", pos_x_mean.expand(total_iso, -1), rotation)
            rotation = torch.repeat_interleave(rotation, num_nodes, dim=0)
            t = torch.repeat_interleave(t, num_nodes, dim=0)
            pos_x = torch.einsum("kj,kij->ki", pos_x, rotation) + t
            loss = (pos_y - pos_x).norm(dim=-1, keepdim=True).view(-1, num_nodes,).mean(-1)
            return torch.argmin(loss)

    @staticmethod
    def update_iso(pos_y, pos_x, batch):
        with torch.no_grad():
            pre_nodes = 0
            num_nodes = batch.n_nodes
            isomorphisms = batch.isomorphisms
            new_idx_x = []
            for i in range(batch.num_graphs):
                cur_num_nodes = num_nodes[i]
                current_isomorphisms = [
                    torch.LongTensor(iso).to(pos_x.device) for iso in isomorphisms[i]
                ]
                if len(current_isomorphisms) == 1:
                    new_idx_x.append(current_isomorphisms[0] + pre_nodes)
                else:
                    pos_y_i = pos_y[pre_nodes : pre_nodes + cur_num_nodes]
                    pos_x_i = pos_x[pre_nodes : pre_nodes + cur_num_nodes]
                    pos_y_mean = torch.mean(pos_y_i, dim=0, keepdim=True)
                    pos_x_mean = torch.mean(pos_x_i, dim=0, keepdim=True)
                    pos_x_list = []

                    for iso in current_isomorphisms:
                        pos_x_list.append(torch.index_select(pos_x_i, 0, iso))
                    total_iso = len(pos_x_list)
                    pos_y_i = pos_y_i.repeat(total_iso, 1)
                    pos_x_i = torch.cat(pos_x_list, dim=0)
                    min_idx = GNN.alignment_loss_iso_onegraph(
                        pos_y_i,
                        pos_x_i,
                        pos_y_mean,
                        pos_x_mean,
                        num_nodes=cur_num_nodes,
                        total_iso=total_iso,
                    )
                    new_idx_x.append(current_isomorphisms[min_idx.item()] + pre_nodes)
                pre_nodes += cur_num_nodes

            return torch.cat(new_idx_x, dim=0)

    @staticmethod
    def alignment(pos_y, pos_x, batch):
        with torch.no_grad():
            num_nodes = batch.n_nodes
            total_nodes = pos_y.shape[0]
            num_graphs = batch.num_graphs
            pos_y_mean = global_mean_pool(pos_y, batch.batch)
            pos_x_mean = global_mean_pool(pos_x, batch.batch)
            y = pos_y - torch.repeat_interleave(pos_y_mean, num_nodes, dim=0)
            x = pos_x - torch.repeat_interleave(pos_x_mean, num_nodes, dim=0)
            a = y + x
            b = y - x
            a = a.view(total_nodes, 1, 3)
            b = b.view(total_nodes, 3, 1)
            tmp0 = torch.cat(
                [b.new_zeros((1, 1, 1)).expand(total_nodes, -1, -1), -b.permute(0, 2, 1)], dim=-1
            )
            eye = torch.eye(3).to(a).unsqueeze(0).expand(total_nodes, -1, -1)
            a = a.expand(-1, 3, -1)
            tmp1 = torch.cross(eye, a, dim=-1)
            tmp1 = torch.cat([b, tmp1], dim=-1)
            tmp = torch.cat([tmp0, tmp1], dim=1)
            tmpb = torch.bmm(tmp.permute(0, 2, 1), tmp).view(total_nodes, -1)
            tmpb = global_mean_pool(tmpb, batch.batch).view(num_graphs, 4, 4)
            w, v = torch.linalg.eigh(tmpb)
            min_rmsd = w[:, 0]
            min_q = v[:, :, 0]
            rotation = GNN.quaternion_to_rotation_matrix(min_q)
            t = pos_y_mean - torch.einsum("kj,kij->ki", pos_x_mean, rotation)
            rotation = torch.repeat_interleave(rotation, num_nodes, dim=0)
            t = torch.repeat_interleave(t, num_nodes, dim=0)
        pos_x = torch.einsum("kj,kij->ki", pos_x, rotation) + t
        return pos_x

    def position_Langevin_Dynamic(self, batch, pos_init, args):
        sigmas = torch.as_tensor(
            np.exp(np.linspace(np.log(args.sigma_begin), np.log(args.sigma_end), args.noise_level,))
        ).to(pos_init)
        pos = pos_init
        for sigma in sigmas:
            step_size = args.noise_lr * (sigma / sigmas[-1]) ** 2
            for step in range(args.noise_steps):
                score_pos = self.get_score(batch, pos, args)
                score_pos = clip_norm(score_pos, 1000)
                noise = torch.randn_like(pos) * torch.sqrt(step_size * 2)
                pos = (pos + step_size * score_pos + noise).detach()

        return pos

    def get_score(self, batch, pos, args):
        assert self.training
        (x, edge_index, edge_attr, node_batch, num_nodes, num_edges, num_graphs,) = (
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
            batch.n_nodes,
            batch.n_edges,
            batch.num_graphs,
        )

        onehot_x = one_hot_atoms(x)
        onehot_edge_attr = one_hot_bonds(edge_attr)

        graph_idx = torch.arange(num_graphs).to(x.device)
        edge_batch = torch.repeat_interleave(graph_idx, num_edges, dim=0)

        x_embed = self.encoder_node(onehot_x)
        edge_attr_embed = self.encoder_edge(onehot_edge_attr)
        u_embed = self.global_init.expand(num_graphs, -1)

        # prior conf
        with torch.no_grad():

            cur_pos = x_embed.new_zeros((x_embed.size(0), 3)).uniform_(-1, 1)
            pos_list = []

            x = x_embed
            edge_attr = edge_attr_embed
            u = u_embed
            for i, layer in enumerate(self.prior_conf_gnns):
                extended_x, extended_edge_attr = self.extend_x_edge(
                    cur_pos, x, edge_attr, edge_index
                )
                x_1, edge_attr_1, u_1 = layer(
                    extended_x,
                    edge_index,
                    extended_edge_attr,
                    u,
                    edge_batch,
                    node_batch,
                    num_nodes,
                    num_edges,
                )
                x = F.dropout(x_1, p=self.dropout, training=self.training) + x
                edge_attr = (
                    F.dropout(edge_attr_1, p=self.dropout, training=self.training) + edge_attr
                )
                u = F.dropout(u_1, p=self.dropout, training=self.training) + u
                if self.pred_pos_residual:
                    delta_pos = self.prior_conf_pos[i](x)
                    cur_pos = self.move2origin(cur_pos + delta_pos, batch)
                else:
                    cur_pos = self.prior_conf_pos[i](x)
                    cur_pos = self.move2origin(cur_pos, batch)
                cur_pos = self.random_augmentation(cur_pos, batch)
                pos_list.append(cur_pos)

            if args.score_prior:
                pass
            else:
                z = torch.randn_like(u_embed)

            prior_output = [x, edge_attr, u]
            z_decoder = torch.repeat_interleave(z, num_nodes, dim=0)

            if self.reuse_prior:
                x, edge_attr, u = prior_output
            else:
                x, edge_attr, u = x_embed, edge_attr_embed, u_embed

            cur_pos = pos_list[-1]
            pos_list = []

            for i, layer in enumerate(self.decoder_gnns):
                if i == len(self.decoder_gnns) - 1:
                    cycle = self.cycle
                else:
                    cycle = 1
                for _ in range(cycle):
                    extended_x, extended_edge_attr = self.extend_x_edge(
                        cur_pos, x + z_decoder, edge_attr, edge_index
                    )
                    x_1, edge_attr_1, u_1 = layer(
                        extended_x,
                        edge_index,
                        extended_edge_attr,
                        u,
                        edge_batch,
                        node_batch,
                        num_nodes,
                        num_edges,
                    )
                    x = F.dropout(x_1, p=self.dropout, training=self.training) + x
                    edge_attr = (
                        F.dropout(edge_attr_1, p=self.dropout, training=self.training) + edge_attr
                    )
                    u = F.dropout(u_1, p=self.dropout, training=self.training) + u
                    if self.pred_pos_residual:
                        delta_pos = self.decoder_pos[i](x)
                        cur_pos = self.move2origin(cur_pos + delta_pos, batch)
                    else:
                        cur_pos = self.decoder_pos[i](x)
                        cur_pos = self.move2origin(cur_pos, batch)
                    cur_pos = self.random_augmentation(cur_pos, batch)
                    pos_list.append(cur_pos)
                    if self.sg_pos:
                        cur_pos = cur_pos.detach()

            score_decoder = (
                -(pos - self.alignment(pos, pos_list[-1], batch)) / args.decoder_std ** 2
            )

        x = x_embed
        edge_attr = edge_attr_embed
        u = u_embed
        pos.requires_grad = True
        cur_pos = self.move2origin(pos, batch)
        cur_pos = get_random_rotation_3d(cur_pos)
        for i, layer in enumerate(self.encoder_gnns):
            extended_x, extended_edge_attr = self.extend_x_edge(cur_pos, x, edge_attr, edge_index)
            x_1, edge_attr_1, u_1 = layer(
                extended_x,
                edge_index,
                extended_edge_attr,
                u,
                edge_batch,
                node_batch,
                num_nodes,
                num_edges,
            )
            x = F.dropout(x_1, p=self.dropout, training=self.training) + x
            edge_attr = F.dropout(edge_attr_1, p=self.dropout, training=self.training) + edge_attr
            u = F.dropout(u_1, p=self.dropout, training=self.training) + u

        if self.use_global:
            aggregated_feat = u
        else:
            aggregated_feat = self.pooling(x, node_batch)

        latent = self.encoder_head(aggregated_feat)
        latent_mean, latent_logstd = torch.chunk(latent, chunks=2, dim=-1)

        logq = -latent_logstd - 0.5 * (latent_mean - z) ** 2 / (torch.exp(latent_logstd)) ** 2
        score_encoder = torch.autograd.grad(logq.sum(), pos)[0]

        return (score_decoder - score_encoder).detach()


def one_hot_atoms(atoms):
    vocab_sizes = get_atom_feature_dims()
    one_hots = []
    for i in range(atoms.shape[1]):
        one_hots.append(
            F.one_hot(atoms[:, i], num_classes=vocab_sizes[i]).to(atoms.device).to(torch.float32)
        )
    return torch.cat(one_hots, dim=1)


def one_hot_bonds(bonds):
    vocab_sizes = get_bond_feature_dims()
    one_hots = []
    for i in range(bonds.shape[1]):
        one_hots.append(
            F.one_hot(bonds[:, i], num_classes=vocab_sizes[i]).to(bonds.device).to(torch.float32)
        )
    return torch.cat(one_hots, dim=1)

