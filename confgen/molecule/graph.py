from confgen.molecule.features import (
    atom_to_feature_vector,
    bond_to_feature_vector,
    extendedbond_to_feature_vector,
)
from rdkit import Chem
import numpy as np
from rdkit.Chem import Mol
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import torch


def rdk2graphedge(mol: Mol):

    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))

    x = np.array(atom_features_list, dtype=np.int64)

    num_bond_features = 3
    if len(mol.GetBonds()) > 0:
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = bond_to_feature_vector(bond)
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        edge_index = np.array(edges_list, dtype=np.int64).T
        # edge_attr = np.array(edge_features_list, dtype=np.int64)

        # add higher order edge
        edge_index_tensor = torch.as_tensor(edge_index)
        adj = to_dense_adj(edge_index_tensor).squeeze(0)
        adj_mats = [
            torch.eye(adj.size(0), dtype=torch.long),
            binarize(adj + torch.eye(adj.size(0), dtype=torch.long)),
        ]
        for i in range(2, 4):
            adj_mats.append(binarize(adj_mats[i - 1] @ adj_mats[1]))
        order_mats = []
        for i in range(1, 4):
            order_mats.append(adj_mats[i] - adj_mats[i - 1])
        order_mats = [dense_to_sparse(x)[0] for x in order_mats]

        for row, col in zip(order_mats[1][0].tolist(), order_mats[1][1].tolist()):
            edges_list.append((row, col))
            edge_features_list.append(extendedbond_to_feature_vector(2))

        for row, col in zip(order_mats[2][0].tolist(), order_mats[2][1].tolist()):
            edges_list.append((row, col))
            edge_features_list.append(extendedbond_to_feature_vector(3))

        edge_index = np.array(edges_list, dtype=np.int64).T
        edge_attr = np.array(edge_features_list, dtype=np.int64)

    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

    graph = dict()
    graph["edge_index"] = edge_index
    graph["edge_attr"] = edge_attr
    graph["node_feat"] = x
    graph["num_nodes"] = len(x)
    graph["n_nodes"] = len(x)
    graph["n_edges"] = len(edge_attr)

    return graph


def rdk2graph(mol: Mol):

    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))

    x = np.array(atom_features_list, dtype=np.int64)

    num_bond_features = 3
    if len(mol.GetBonds()) > 0:
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = bond_to_feature_vector(bond)
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        edge_index = np.array(edges_list, dtype=np.int64).T
        edge_attr = np.array(edge_features_list, dtype=np.int64)

        # # add higher order edge
        # edge_index_tensor = torch.as_tensor(edge_index)
        # adj = to_dense_adj(edge_index_tensor).squeeze(0)
        # adj_mats = [
        #     torch.eye(adj.size(0), dtype=torch.long),
        #     binarize(adj + torch.eye(adj.size(0), dtype=torch.long)),
        # ]
        # for i in range(2, 4):
        #     adj_mats.append(binarize(adj_mats[i - 1] @ adj_mats[1]))
        # order_mats = []
        # for i in range(1, 4):
        #     order_mats.append(adj_mats[i] - adj_mats[i - 1])
        # order_mats = [dense_to_sparse(x)[0] for x in order_mats]

        # for row, col in zip(order_mats[1][0].tolist(), order_mats[1][1].tolist()):
        #     edges_list.append((row, col))
        #     edge_features_list.append(extendedbond_to_feature_vector(2))

        # for row, col in zip(order_mats[2][0].tolist(), order_mats[2][1].tolist()):
        #     edges_list.append((row, col))
        #     edge_features_list.append(extendedbond_to_feature_vector(3))

        # edge_index = np.array(edges_list, dtype=np.int64).T
        # edge_attr = np.array(edge_features_list, dtype=np.int64)

    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

    n_src = list()
    n_tgt = list()
    for atom in mol.GetAtoms():
        n_ids = [n.GetIdx() for n in atom.GetNeighbors()]
        if len(n_ids) > 1:
            n_src.append(atom.GetIdx())
            n_tgt.append(n_ids)
    nums_neigh = len(n_src)
    nei_src_index = np.array(n_src, dtype=np.int64).reshape(1, -1)
    nei_tgt_index = np.zeros((6, nums_neigh), dtype=np.int64)
    nei_tgt_mask = np.ones((6, nums_neigh), dtype=np.bool)

    for i, n_ids in enumerate(n_tgt):
        nei_tgt_index[: len(n_ids), i] = n_ids
        nei_tgt_mask[: len(n_ids), i] = False

    graph = dict()
    graph["edge_index"] = edge_index
    graph["edge_attr"] = edge_attr
    graph["node_feat"] = x
    graph["num_nodes"] = len(x)
    graph["n_nodes"] = len(x)
    graph["n_edges"] = len(edge_attr)

    graph["nei_src_index"] = nei_src_index
    graph["nei_tgt_index"] = nei_tgt_index
    graph["nei_tgt_mask"] = nei_tgt_mask

    return graph


def binarize(x):
    return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))


if __name__ == "__main__":
    rdk2graph(Chem.MolFromSmiles(r"[H]c1oc([H])c(C#CC#N)c1[H]"))
