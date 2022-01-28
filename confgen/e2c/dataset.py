from genericpath import exists
import numpy as np
import random
import os
import json
from tqdm import tqdm
import pickle
from rdkit import Chem
from rdkit.Chem.rdchem import Mol, HybridizationType
import torch
from confgen import utils
from torch_geometric.data import InMemoryDataset, Data
from torch_sparse import SparseTensor
import re
import confgen
from ..molecule.graph import rdk2graph, rdk2graphedge
import copy
from rdkit.Chem.rdmolops import RemoveHs
from confgen.molecule.gt import isomorphic_core


class PygGeomDataset(InMemoryDataset):
    def __init__(
        self,
        root="dataset",
        rdk2graph=rdk2graph,
        transform=None,
        pre_transform=None,
        dataset="qm9",
        base_path="/data/code/ConfGF/dataset",
        seed=None,
        extend_edge=False,
        data_split="cgcf",
        remove_hs=False,
    ):
        self.original_root = root
        self.rdk2graph = rdk2graph
        if seed == None:
            self.seed = 2021
        else:
            self.seed = seed
        assert dataset in ["qm9", "drugs", "iso17"]
        self.folder = os.path.join(root, f"geom_{dataset}_{data_split}")
        if extend_edge:
            self.rdk2graph = rdk2graphedge
            self.folder = os.path.join(root, f"geom_{dataset}_{data_split}_ee")

        if remove_hs:
            self.folder = os.path.join(root, f"geom_{dataset}_{data_split}_rh_ext_gt")
        else:
            self.folder = os.path.join(root, f"geom_{dataset}_{data_split}_ext_gt")

        self.base_path = base_path
        self.dataset_name = dataset
        self.data_split = data_split
        self.remove_hs = remove_hs

        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # no error, since download function will not download anything
        return "data.csv.gz"

    @property
    def processed_file_names(self):
        return "geometric_data_processed.pt"

    def download(self):
        if os.path.exists(self.processed_paths[0]):
            return
        else:
            assert os.path.exists(self.base_path)

    def process(self):
        print("Converting pickle files into graphs...")
        assert self.dataset_name in ["qm9", "drugs", "iso17"]
        if self.data_split == "cgcf":
            self.process_cgcf()
            return
        if self.data_split == "confgf":
            self.process_confgf()
            return
        if self.data_split == "default":
            self.process_default()
            return
        summary_path = os.path.join(self.base_path, f"summary_{self.dataset_name}.json")
        with open(summary_path, "r") as src:
            summ = json.load(src)

        pickle_path_list = []
        for smiles, meta_mol in tqdm(summ.items()):
            u_conf = meta_mol.get("uniqueconfs")
            if u_conf is None:
                continue
            pickle_path = meta_mol.get("pickle_path")
            if pickle_path is None:
                continue
            if "." in smiles:
                continue
            pickle_path_list.append(pickle_path)

        data_list = []
        num_mols = 0
        num_confs = 0
        bad_case = 0

        random.seed(19970327)
        random.shuffle(pickle_path_list)
        train_size = int(len(pickle_path_list) * 0.8)
        valid_size = int(len(pickle_path_list) * 0.9)
        train_idx = []
        valid_idx = []
        test_idx = []

        for i, pickle_path in enumerate(tqdm(pickle_path_list)):
            if self.dataset_name in ["drugs"]:
                if i < train_size:
                    if len(train_idx) >= 2000000:
                        continue
                elif i < valid_size:
                    if len(valid_idx) >= 100000:
                        continue
                else:
                    if len(test_idx) >= 100000:
                        continue

            with open(os.path.join(self.base_path, pickle_path), "rb") as src:
                mol = pickle.load(src)
            if mol.get("uniqueconfs") != len(mol.get("conformers")):
                bad_case += 1
                continue
            if mol.get("uniqueconfs") <= 0:
                bad_case += 1
                continue
            if mol.get("conformers")[0]["rd_mol"].GetNumBonds() < 1:
                bad_case += 1
                continue
            if "." in Chem.MolToSmiles(mol.get("conformers")[0]["rd_mol"]):
                bad_case += 1
                continue
            num_mols += 1

            for conf_meta in mol.get("conformers"):
                if self.remove_hs:
                    try:
                        new_mol = RemoveHs(conf_meta["rd_mol"])
                    except Exception:
                        continue
                else:
                    new_mol = conf_meta["rd_mol"]
                graph = self.rdk2graph(new_mol)

                assert len(graph["edge_attr"]) == graph["edge_index"].shape[1]
                assert len(graph["node_feat"]) == graph["num_nodes"]

                data = Data()
                data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
                data.edge_attr = torch.from_numpy(graph["edge_attr"]).to(torch.int64)
                data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
                data.n_nodes = graph["n_nodes"]
                data.n_edges = graph["n_edges"]
                data.pos = torch.from_numpy(new_mol.GetConformer(0).GetPositions()).to(torch.float)

                data.lowestenergy = torch.as_tensor([mol.get("lowestenergy")]).to(torch.float)
                data.energy = torch.as_tensor([conf_meta["totalenergy"]]).to(torch.float)
                data.rd_mol = copy.deepcopy(new_mol)
                data.isomorphisms = isomorphic_core(new_mol)

                data.nei_src_index = torch.from_numpy(graph["nei_src_index"]).to(torch.int64)
                data.nei_tgt_index = torch.from_numpy(graph["nei_tgt_index"]).to(torch.int64)
                data.nei_tgt_mask = torch.from_numpy(graph["nei_tgt_mask"]).to(torch.bool)

                if i < train_size:
                    train_idx.append(len(data_list))
                elif i < valid_size:
                    valid_idx.append(len(data_list))
                else:
                    test_idx.append(len(data_list))

                data_list.append(data)

                num_confs += 1

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        print("Saving...")
        print(f"num mols {num_mols} num confs {num_confs} num bad cases {bad_case}")
        torch.save((data, slices), self.processed_paths[0])
        os.makedirs(os.path.join(self.root, "split"), exist_ok=True)
        torch.save(
            {
                "train": torch.tensor(train_idx, dtype=torch.long),
                "valid": torch.tensor(valid_idx, dtype=torch.long),
                "test": torch.tensor(test_idx, dtype=torch.long),
            },
            os.path.join(self.root, "split", "split_dict.pt"),
        )

    def get_idx_split(self):
        path = os.path.join(self.root, "split")
        if os.path.isfile(os.path.join(path, "split_dict.pt")):
            return torch.load(os.path.join(path, "split_dict.pt"))

    def process_cgcf(self):
        valid_conformation = 0
        train_idx = []
        valid_idx = []
        test_idx = []
        data_list = []
        bad_case = 0
        for subset in ["train", "val", "test"]:
            pkl_fn = os.path.join(
                self.base_path, self.dataset_name, f"{subset}_{self.dataset_name.upper()}.pkl"
            )
            with open(pkl_fn, "rb") as src:
                mol_list = pickle.load(src)

            for mol in tqdm(mol_list):
                if "." in Chem.MolToSmiles(mol):
                    bad_case += 1
                    continue
                if mol.GetNumBonds() < 1:
                    bad_case += 1
                    continue
                graph = self.rdk2graph(mol)
                assert len(graph["edge_attr"]) == graph["edge_index"].shape[1]
                assert len(graph["node_feat"]) == graph["num_nodes"]

                data = Data()
                data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
                data.edge_attr = torch.from_numpy(graph["edge_attr"]).to(torch.int64)
                data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
                data.n_nodes = graph["n_nodes"]
                data.n_edges = graph["n_edges"]
                data.pos = torch.from_numpy(mol.GetConformer(0).GetPositions()).to(torch.float)

                data.rd_mol = copy.deepcopy(mol)

                if subset == "train":
                    train_idx.append(valid_conformation)
                elif subset == "val":
                    valid_idx.append(valid_conformation)
                else:
                    test_idx.append(valid_conformation)
                valid_conformation += 1

                data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        print("Saving...")
        print(f"num confs {valid_conformation} num bad cases {bad_case}")
        torch.save((data, slices), self.processed_paths[0])
        os.makedirs(os.path.join(self.root, "split"), exist_ok=True)
        torch.save(
            {
                "train": torch.tensor(train_idx, dtype=torch.long),
                "valid": torch.tensor(valid_idx, dtype=torch.long),
                "test": torch.tensor(test_idx, dtype=torch.long),
            },
            os.path.join(self.root, "split", "split_dict.pt"),
        )

    def process_confgf(self):
        valid_conformation = 0
        train_idx = []
        valid_idx = []
        test_idx = []
        data_list = []
        bad_case = 0
        file_name = ["train_data_40k", "val_data_5k", "test_data_200"]
        if self.dataset_name == "drugs":
            file_name[0] = "train_data_39k"
        if self.dataset_name == "iso17":
            file_name = ["iso17_split-0_train_processed", "iso17_split-0_test_processed"]
        for subset in file_name:
            pkl_fn = os.path.join(self.base_path, f"{subset}.pkl")
            with open(pkl_fn, "rb") as src:
                mol_list = pickle.load(src)
            mol_list = [x.rdmol for x in mol_list]

            for mol in tqdm(mol_list):
                if self.remove_hs:
                    try:
                        mol = RemoveHs(mol)
                    except Exception:
                        continue
                if "." in Chem.MolToSmiles(mol):
                    bad_case += 1
                    continue
                if mol.GetNumBonds() < 1:
                    bad_case += 1
                    continue
                graph = self.rdk2graph(mol)
                assert len(graph["edge_attr"]) == graph["edge_index"].shape[1]
                assert len(graph["node_feat"]) == graph["num_nodes"]

                data = CustomData()
                data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
                data.edge_attr = torch.from_numpy(graph["edge_attr"]).to(torch.int64)
                data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
                data.n_nodes = graph["n_nodes"]
                data.n_edges = graph["n_edges"]
                data.pos = torch.from_numpy(mol.GetConformer(0).GetPositions()).to(torch.float)

                data.rd_mol = copy.deepcopy(mol)
                data.isomorphisms = isomorphic_core(mol)

                data.nei_src_index = torch.from_numpy(graph["nei_src_index"]).to(torch.int64)
                data.nei_tgt_index = torch.from_numpy(graph["nei_tgt_index"]).to(torch.int64)
                data.nei_tgt_mask = torch.from_numpy(graph["nei_tgt_mask"]).to(torch.bool)

                if "train" in subset:
                    train_idx.append(valid_conformation)
                elif "val" in subset:
                    valid_idx.append(valid_conformation)
                else:
                    test_idx.append(valid_conformation)
                valid_conformation += 1

                data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        print("Saving...")
        print(f"num confs {valid_conformation} num bad cases {bad_case}")
        torch.save((data, slices), self.processed_paths[0])
        os.makedirs(os.path.join(self.root, "split"), exist_ok=True)
        if len(valid_idx) == 0:
            valid_idx = train_idx[:6400]
        torch.save(
            {
                "train": torch.tensor(train_idx, dtype=torch.long),
                "valid": torch.tensor(valid_idx, dtype=torch.long),
                "test": torch.tensor(test_idx, dtype=torch.long),
            },
            os.path.join(self.root, "split", "split_dict.pt"),
        )

    def process_default(self):
        data_list = []
        file_name = os.path.join(self.base_path, f"geom_{self.dataset_name}_default_rh.pt")
        mol_list, split_dict = torch.load(file_name)
        for mol in tqdm(mol_list):
            graph = self.rdk2graph(mol)
            assert len(graph["edge_attr"]) == graph["edge_index"].shape[1]
            assert len(graph["node_feat"]) == graph["num_nodes"]

            data = CustomData()
            data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
            data.edge_attr = torch.from_numpy(graph["edge_attr"]).to(torch.int64)
            data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
            data.n_nodes = graph["n_nodes"]
            data.n_edges = graph["n_edges"]
            data.pos = torch.from_numpy(mol.GetConformer(0).GetPositions()).to(torch.float)

            data.rd_mol = copy.deepcopy(mol)
            data.isomorphisms = isomorphic_core(mol)

            data.nei_src_index = torch.from_numpy(graph["nei_src_index"]).to(torch.int64)
            data.nei_tgt_index = torch.from_numpy(graph["nei_tgt_index"]).to(torch.int64)
            data.nei_tgt_mask = torch.from_numpy(graph["nei_tgt_mask"]).to(torch.bool)

            data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])
        os.makedirs(os.path.join(self.root, "split"), exist_ok=True)
        torch.save(
            split_dict, os.path.join(self.root, "split", "split_dict.pt"),
        )


class CustomData(Data):
    def __cat_dim__(self, key, value):
        if isinstance(value, SparseTensor):
            return (0, 1)
        elif bool(re.search("(index|face|nei_tgt_mask)", key)):
            return -1
        return 0
