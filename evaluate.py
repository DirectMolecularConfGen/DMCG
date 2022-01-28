import copy
import os
import argparse
from rdkit import Chem
import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import random
from confgen.e2c.dataset import PygGeomDataset
from confgen.model.gnn import GNN
from torch.optim.lr_scheduler import LambdaLR
from confgen.utils.utils import (
    WarmCosine,
    set_rdmol_positions,
    get_best_rmsd,
    evaluate_distance,
)
import io
import json
from collections import defaultdict
import multiprocessing
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule


def train(model, device, loader, optimizer, scheduler, args):
    model.train()
    loss_accum_dict = defaultdict(float)
    pbar = tqdm(loader, desc="Iteration")
    for step, batch in enumerate(pbar):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            atom_pred_list, extra_output = model(batch)
            optimizer.zero_grad()

            loss, loss_dict = model.compute_loss(atom_pred_list, extra_output, batch, args)
            loss.backward()
            optimizer.step()
            scheduler.step()

            for k, v in loss_dict.items():
                loss_accum_dict[k] += v.detach().item()

            if step % args.log_interval == 0:
                description = f"Iteration loss: {loss_accum_dict['loss'] / (step + 1):6.4f} lr: {scheduler.get_last_lr()[0]:.5e}"
                # for k in loss_accum_dict.keys():
                #     description += f" {k}: {loss_accum_dict[k]/(step+1):6.4f}"

                pbar.set_description(description)

    for k in loss_accum_dict.keys():
        loss_accum_dict[k] /= step + 1
    return loss_accum_dict


def get_rmsd_min(inputargs):
    mols, use_ff, threshold = inputargs
    gen_mols, ref_mols = mols
    rmsd_mat = np.zeros([len(ref_mols), len(gen_mols)], dtype=np.float32)
    for i, gen_mol in enumerate(gen_mols):
        gen_mol_c = copy.deepcopy(gen_mol)
        if use_ff:
            MMFFOptimizeMolecule(gen_mol_c)
        for j, ref_mol in enumerate(ref_mols):
            ref_mol_c = copy.deepcopy(ref_mol)
            rmsd_mat[j, i] = get_best_rmsd(gen_mol_c, ref_mol_c)
    rmsd_mat_min = rmsd_mat.min(-1)
    return (rmsd_mat_min <= threshold).mean(), rmsd_mat_min.mean()


def evaluate(model, device, loader, args):
    model.eval()
    mol_labels = []
    mol_preds = []
    for batch in tqdm(loader, desc="Iteration"):
        batch = batch.to(device)
        with torch.no_grad():
            pred, _ = model(batch, sample=True)
        pred = pred[-1]
        batch_size = batch.num_graphs
        n_nodes = batch.n_nodes.tolist()
        pre_nodes = 0
        for i in range(batch_size):
            mol_labels.append(batch.rd_mol[i])
            mol_preds.append(
                set_rdmol_positions(batch.rd_mol[i], pred[pre_nodes : pre_nodes + n_nodes[i]])
            )
            pre_nodes += n_nodes[i]

        with torch.no_grad():
            pred, _ = model(batch, sample=True)
        pred = pred[-1]
        batch_size = batch.num_graphs
        n_nodes = batch.n_nodes.tolist()
        pre_nodes = 0
        for i in range(batch_size):
            mol_preds.append(
                set_rdmol_positions(batch.rd_mol[i], pred[pre_nodes : pre_nodes + n_nodes[i]])
            )
            pre_nodes += n_nodes[i]

    # for gen_mol, ref_mol in zip(mol_preds, mol_labels):
    #     try:
    #         rmsd_list.append(get_best_rmsd(gen_mol, ref_mol))
    #     except Exception as e:
    #         continue

    smiles2pairs = dict()
    for gen_mol in mol_preds:
        smiles = Chem.MolToSmiles(gen_mol)
        if smiles not in smiles2pairs:
            smiles2pairs[smiles] = [[gen_mol]]
        else:
            smiles2pairs[smiles][0].append(gen_mol)
    for ref_mol in mol_labels:
        smiles = Chem.MolToSmiles(ref_mol)
        if len(smiles2pairs[smiles]) == 1:
            smiles2pairs[smiles].append([ref_mol])
        else:
            smiles2pairs[smiles][1].append(ref_mol)

    del_smiles = []
    for smiles in smiles2pairs.keys():
        if len(smiles2pairs[smiles][1]) < 50 or len(smiles2pairs[smiles][1]) > 500:
            del_smiles.append(smiles)
    for smiles in del_smiles:
        del smiles2pairs[smiles]

    cov_list = []
    mat_list = []
    pool = multiprocessing.Pool(args.workers)

    def input_args():
        for smiles in smiles2pairs.keys():
            yield smiles2pairs[smiles], args.use_ff, 0.5 if args.dataset_name == "qm9" else 1.25

    for res in tqdm(pool.imap(get_rmsd_min, input_args(), chunksize=10), total=len(smiles2pairs)):
        cov_list.append(res[0])
        mat_list.append(res[1])
    print(f"cov mean {np.mean(cov_list)} med {np.median(cov_list)}")
    print(f"mat mean {np.mean(mat_list)} med {np.median(mat_list)}")
    return np.mean(cov_list), np.mean(mat_list)


def evaluate_one(model, device, loader):
    model.eval()
    mol_labels = []
    mol_preds = []
    for batch in tqdm(loader, desc="Iteration"):
        batch = batch.to(device)
        with torch.no_grad():
            pred, _ = model(batch)
        pred = pred[-1]
        batch_size = batch.num_graphs
        n_nodes = batch.n_nodes.tolist()
        pre_nodes = 0
        for i in range(batch_size):
            mol_labels.append(batch.rd_mol[i])
            mol_preds.append(
                set_rdmol_positions(batch.rd_mol[i], pred[pre_nodes : pre_nodes + n_nodes[i]])
            )
            pre_nodes += n_nodes[i]

    rmsd_list = []
    for gen_mol, ref_mol in zip(mol_preds, mol_labels):
        try:
            a = get_best_rmsd(gen_mol, ref_mol)
            if a > 100:
                print(a)
                print(Chem.MolToSmiles(ref_mol))
            rmsd_list.append(a)
        except Exception as e:
            continue

    return np.mean(rmsd_list)


def evaluate_iso17(model, device, loader, args):
    model.eval()
    mol_labels = []
    mol_preds = []
    for batch in tqdm(loader, desc="Iteration"):
        batch = batch.to(device)
        with torch.no_grad():
            pred, _ = model(batch, sample=True)
        pred = pred[-1]
        batch_size = batch.num_graphs
        n_nodes = batch.n_nodes.tolist()
        pre_nodes = 0
        for i in range(batch_size):
            mol_labels.append(batch.rd_mol[i])
            mol_preds.append(
                set_rdmol_positions(batch.rd_mol[i], pred[pre_nodes : pre_nodes + n_nodes[i]])
            )
            pre_nodes += n_nodes[i]

    smiles2pairs = dict()
    for gen_mol in mol_preds:
        smiles = Chem.MolToSmiles(gen_mol)
        if smiles not in smiles2pairs:
            smiles2pairs[smiles] = [[gen_mol]]
        else:
            smiles2pairs[smiles][0].append(gen_mol)

    for ref_mol in mol_labels:
        smiles = Chem.MolToSmiles(ref_mol)
        if len(smiles2pairs[smiles]) == 1:
            smiles2pairs[smiles].append([ref_mol])
        else:
            smiles2pairs[smiles][1].append(ref_mol)

    del_smiles = []
    for smiles in smiles2pairs.keys():
        if len(smiles2pairs[smiles][1]) < 1000:
            del_smiles.append(smiles)
    for smiles in del_smiles:
        del smiles2pairs[smiles]

    s_mmd_all = []
    p_mmd_all = []
    a_mmd_all = []

    print(f"{len(smiles2pairs)} mols")
    for smiles in smiles2pairs.keys():
        gen_mols, ref_mols = smiles2pairs[smiles]

        stats_single, stats_pair, stats_all = evaluate_distance(gen_mols, ref_mols)
        s_mmd_all += [e["mmd"] for e in stats_single]
        p_mmd_all += [e["mmd"] for e in stats_pair]
        a_mmd_all.append(stats_all["mmd"])

    print(
        "SingleDist | Mean: %.4f | Median: %.4f | Min: %.4f | Max: %.4f"
        % (np.mean(s_mmd_all), np.median(s_mmd_all), np.min(s_mmd_all), np.max(s_mmd_all))
    )
    print(
        "PairDist | Mean: %.4f | Median: %.4f | Min: %.4f | Max: %.4f"
        % (np.mean(p_mmd_all), np.median(p_mmd_all), np.min(p_mmd_all), np.max(p_mmd_all))
    )
    print(
        "AllDist | Mean: %.4f | Median: %.4f | Min: %.4f | Max: %.4f"
        % (np.mean(a_mmd_all), np.median(a_mmd_all), np.min(a_mmd_all), np.max(a_mmd_all))
    )


def evaluate_score(model, device, loader, args):

    # one step
    if False:
        model.eval()
        mol_labels = []
        mol_preds = []
        for batch in tqdm(loader, desc="Iteration"):
            batch = batch.to(device)
            with torch.no_grad():
                pred, _ = model(batch, sample=True)
            pred = pred[-1]
            batch_size = batch.num_graphs
            n_nodes = batch.n_nodes.tolist()
            pre_nodes = 0
            for i in range(batch_size):
                mol_labels.append(batch.rd_mol[i])
                mol_preds.append(
                    set_rdmol_positions(batch.rd_mol[i], pred[pre_nodes : pre_nodes + n_nodes[i]])
                )
                pre_nodes += n_nodes[i]

            with torch.no_grad():
                pred, _ = model(batch, sample=True)
            pred = pred[-1]
            batch_size = batch.num_graphs
            n_nodes = batch.n_nodes.tolist()
            pre_nodes = 0
            for i in range(batch_size):
                mol_preds.append(
                    set_rdmol_positions(batch.rd_mol[i], pred[pre_nodes : pre_nodes + n_nodes[i]])
                )
                pre_nodes += n_nodes[i]
        smiles2pairs = dict()
        for gen_mol in mol_preds:
            smiles = Chem.MolToSmiles(gen_mol)
            if smiles not in smiles2pairs:
                smiles2pairs[smiles] = [[gen_mol]]
            else:
                smiles2pairs[smiles][0].append(gen_mol)
        for ref_mol in mol_labels:
            smiles = Chem.MolToSmiles(ref_mol)
            if len(smiles2pairs[smiles]) == 1:
                smiles2pairs[smiles].append([ref_mol])
            else:
                smiles2pairs[smiles][1].append(ref_mol)

        del_smiles = []
        for smiles in smiles2pairs.keys():
            if len(smiles2pairs[smiles][1]) < 50 or len(smiles2pairs[smiles][1]) > 500:
                del_smiles.append(smiles)
        for smiles in del_smiles:
            del smiles2pairs[smiles]

        cov_list = []
        mat_list = []
        pool = multiprocessing.Pool(args.workers)

        def input_args():
            for smiles in smiles2pairs.keys():
                yield smiles2pairs[smiles], args.use_ff, 0.5 if args.dataset_name == "qm9" else 1.25

        for res in tqdm(
            pool.imap(get_rmsd_min, input_args(), chunksize=10), total=len(smiles2pairs)
        ):
            cov_list.append(res[0])
            mat_list.append(res[1])
        print(f"prior cov mean {np.mean(cov_list)} med {np.median(cov_list)}")
        print(f"prior mat mean {np.mean(mat_list)} med {np.median(mat_list)}")

    mol_labels = []
    mol_preds = []

    for batch in tqdm(loader, desc="Iteration"):
        batch = batch.to(device)
        model.eval()
        with torch.no_grad():
            pred, _ = model(batch, sample=True)
            prior_pos = pred[-1]
        model.train()
        pred = model.position_Langevin_Dynamic(batch, prior_pos, args)
        batch_size = batch.num_graphs
        n_nodes = batch.n_nodes.tolist()
        pre_nodes = 0
        for i in range(batch_size):
            mol_labels.append(batch.rd_mol[i])
            mol_preds.append(
                set_rdmol_positions(batch.rd_mol[i], pred[pre_nodes : pre_nodes + n_nodes[i]])
            )
            pre_nodes += n_nodes[i]

        model.eval()
        with torch.no_grad():
            pred, _ = model(batch, sample=True)
            prior_pos = pred[-1]
        model.train()
        pred = model.position_Langevin_Dynamic(batch, prior_pos, args)
        batch_size = batch.num_graphs
        n_nodes = batch.n_nodes.tolist()
        pre_nodes = 0
        for i in range(batch_size):
            mol_preds.append(
                set_rdmol_positions(batch.rd_mol[i], pred[pre_nodes : pre_nodes + n_nodes[i]])
            )
            pre_nodes += n_nodes[i]

    smiles2pairs = dict()
    for gen_mol in mol_preds:
        smiles = Chem.MolToSmiles(gen_mol)
        if smiles not in smiles2pairs:
            smiles2pairs[smiles] = [[gen_mol]]
        else:
            smiles2pairs[smiles][0].append(gen_mol)
    for ref_mol in mol_labels:
        smiles = Chem.MolToSmiles(ref_mol)
        if len(smiles2pairs[smiles]) == 1:
            smiles2pairs[smiles].append([ref_mol])
        else:
            smiles2pairs[smiles][1].append(ref_mol)

    del_smiles = []
    for smiles in smiles2pairs.keys():
        if len(smiles2pairs[smiles][1]) < 50 or len(smiles2pairs[smiles][1]) > 500:
            del_smiles.append(smiles)
    for smiles in del_smiles:
        del smiles2pairs[smiles]

    cov_list = []
    mat_list = []
    pool = multiprocessing.Pool(args.workers)

    def input_args():
        for smiles in smiles2pairs.keys():
            yield smiles2pairs[smiles], args.use_ff, 0.5 if args.dataset_name == "qm9" else 1.25

    for res in tqdm(pool.imap(get_rmsd_min, input_args(), chunksize=10), total=len(smiles2pairs)):
        cov_list.append(res[0])
        mat_list.append(res[1])
    print(f"cov mean {np.mean(cov_list)} med {np.median(cov_list)}")
    print(f"mat mean {np.mean(mat_list)} med {np.median(mat_list)}")

    return np.mean(cov_list), np.mean(mat_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--global-reducer", type=str, default="sum")
    parser.add_argument("--node-reducer", type=str, default="sum")
    parser.add_argument("--graph-pooling", type=str, default="sum")
    parser.add_argument("--dropedge-rate", type=float, default=0.1)
    parser.add_argument("--dropnode-rate", type=float, default=0.1)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--decoder-layers", type=int, default=None)
    parser.add_argument("--latent-size", type=int, default=256)
    parser.add_argument("--mlp-hidden-size", type=int, default=1024)
    parser.add_argument("--mlp_layers", type=int, default=2)
    parser.add_argument("--use-layer-norm", action="store_true", default=False)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=1)
    # parser.add_argument("--log-dir", type=str, default="", help="tensorboard log directory")
    parser.add_argument("--checkpoint-dir", type=str, default="")

    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--encoder-dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--layernorm-before", action="store_true", default=False)
    parser.add_argument("--use-bn", action="store_true", default=False)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--use-adamw", action="store_true", default=False)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--period", type=float, default=10)

    parser.add_argument("--base-path", type=str, default="")
    parser.add_argument(
        "--dataset-name", type=str, default="qm9", choices=["qm9", "drugs", "iso17"]
    )
    parser.add_argument("--train-size", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--lr-warmup", action="store_true", default=False)
    parser.add_argument("--enable-tb", action="store_true", default=False)
    parser.add_argument("--aux-loss", type=float, default=0.0)
    parser.add_argument("--train-subset", action="store_true", default=False)
    parser.add_argument("--eval-from", type=str, default=None)
    parser.add_argument(
        "--data-split", type=str, choices=["cgcf", "default", "confgf"], default="default"
    )
    parser.add_argument("--reuse-prior", action="store_true", default=False)
    parser.add_argument("--cycle", type=int, default=1)

    parser.add_argument("--vae-beta", type=float, default=1.0)
    parser.add_argument("--eval-one", action="store_true", default=False)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--extend-edge", action="store_true", default=False)
    parser.add_argument("--use-ff", action="store_true", default=False)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--pred-pos-residual", action="store_true", default=False)
    parser.add_argument("--node-attn", action="store_true", default=False)
    parser.add_argument("--global-attn", action="store_true", default=False)
    parser.add_argument("--shared-decoder", action="store_true", default=False)
    parser.add_argument("--shared-output", action="store_true", default=False)
    parser.add_argument("--sample-beta", type=float, default=1.0)
    parser.add_argument("--remove-hs", action="store_true", default=False)
    parser.add_argument("--prop-pred", action="store_true", default=False)

    parser.add_argument("--score", action="store_true", default=False)
    parser.add_argument("--sigma-begin", type=float, default=10.0)
    parser.add_argument("--sigma-end", type=float, default=0.01)
    parser.add_argument("--noise-level", type=int, default=10)
    parser.add_argument("--noise-steps", type=int, default=100)
    parser.add_argument("--noise-lr", type=float, default=2.4e-6)
    parser.add_argument("--decoder-std", type=float, default=1.0)
    parser.add_argument("--score-prior", action="store_true", default=False)

    args = parser.parse_args()
    print(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    dataset = PygGeomDataset(
        root="dataset",
        dataset=args.dataset_name,
        base_path=args.base_path,
        seed=args.seed,
        extend_edge=args.extend_edge,
        data_split=args.data_split,
        remove_hs=args.remove_hs,
    )

    split_idx = dataset.get_idx_split()
    train_loader = DataLoader(
        dataset[split_idx["train"]]
        if not args.train_subset
        else dataset[split_idx["train"]][:102400],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_loader = DataLoader(
        dataset[split_idx["valid"]],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        dataset[split_idx["test"]],
        batch_size=args.batch_size,
        shuffle=False if not args.score else True,
        num_workers=args.num_workers,
    )

    shared_params = {
        "mlp_hidden_size": args.mlp_hidden_size,
        "mlp_layers": args.mlp_layers,
        "latent_size": args.latent_size,
        "use_layer_norm": args.use_layer_norm,
        "num_message_passing_steps": args.num_layers,
        "global_reducer": args.global_reducer,
        "node_reducer": args.node_reducer,
        "dropedge_rate": args.dropedge_rate,
        "dropnode_rate": args.dropnode_rate,
        "dropout": args.dropout,
        "layernorm_before": args.layernorm_before,
        "encoder_dropout": args.encoder_dropout,
        "use_bn": args.use_bn,
        "vae_beta": args.vae_beta,
        "decoder_layers": args.decoder_layers,
        "reuse_prior": args.reuse_prior,
        "cycle": args.cycle,
        "pred_pos_residual": args.pred_pos_residual,
        "node_attn": args.node_attn,
        "global_attn": args.global_attn,
        "shared_decoder": args.shared_decoder,
        "sample_beta": args.sample_beta,
        "shared_output": args.shared_output,
    }
    model = GNN(**shared_params).to(device)
    if args.eval_from is not None:
        assert os.path.exists(args.eval_from)
        checkpoint = torch.load(args.eval_from, map_location=device)["model_state_dict"]
        cur_state_dict = model.state_dict()
        del_keys = []
        for k in checkpoint.keys():
            if k not in cur_state_dict:
                del_keys.append(k)
        for k in del_keys:
            del checkpoint[k]
        model.load_state_dict(checkpoint)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"#Params: {num_params}")
    if args.use_adamw:
        optimizer = optim.AdamW(
            model.parameters(), lr=args.lr, betas=(0.9, args.beta2), weight_decay=args.weight_decay
        )
    else:
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, betas=(0.9, args.beta2), weight_decay=args.weight_decay
        )

    if not args.lr_warmup:
        scheduler = LambdaLR(optimizer, lambda x: 1.0)
    else:
        lrscheduler = WarmCosine(tmax=len(train_loader) * args.period, warmup=int(4e3))
        scheduler = LambdaLR(optimizer, lambda x: lrscheduler.step(x))

    if args.checkpoint_dir and args.enable_tb:
        tb_writer = SummaryWriter(args.checkpoint_dir)

    train_curve = []
    valid_curve = []
    test_curve = []
    if args.eval_one:
        train_pref = evaluate_one(model, device, train_loader)
        valid_pref = evaluate_one(model, device, valid_loader)
        test_pref = evaluate_one(model, device, test_loader)
        print(f"train {train_pref} valid {valid_pref} test {test_pref}")
        exit(0)

    if args.data_split != "confgf":
        train_pref = evaluate(model, device, train_loader, args)
        valid_pref = evaluate(model, device, valid_loader, args)
    else:
        if args.dataset_name == "iso17":
            evaluate_iso17(model, device, test_loader, args)
            exit(0)
        train_pref = (0, 0)
        valid_pref = (0, 0)
    if args.score:
        test_pref = evaluate_score(model, device, test_loader, args)
    else:
        test_pref = evaluate(model, device, test_loader, args)
    print(f"train {train_pref} valid {valid_pref} test {test_pref}")
    exit(0)

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print("Training...")
        loss_dict = train(model, device, train_loader, optimizer, scheduler, args)
        print("Evaluating...")
        # train_pref = evaluate(model, device, train_loader)
        valid_pref = evaluate(model, device, valid_loader)
        test_pref = evaluate(model, device, test_loader)

        if args.checkpoint_dir:
            print(f"Setting {os.path.basename(os.path.normpath(args.checkpoint_dir))}...")
        # print(f"Train: {train_pref} Validation: {valid_pref} Test: {test_pref}")
        print(f"Train: {loss_dict['loss']} Validation: {valid_pref} Test: {test_pref}")

        # train_curve.append(train_pref)
        valid_curve.append(valid_pref)
        test_curve.append(test_pref)
        if args.checkpoint_dir:
            logs = {"Train": loss_dict["loss"], "Valid": valid_pref, "Test": test_pref}
            with io.open(
                os.path.join(args.checkpoint_dir, "log.txt"), "a", encoding="utf8", newline="\n"
            ) as tgt:
                print(json.dumps(logs), file=tgt)

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, f"checkpoint_{epoch}.pt"))
            if args.enable_tb:
                # tb_writer.add_scalar("evaluation/train", train_pref, epoch)
                tb_writer.add_scalar("evaluation/valid", valid_pref, epoch)
                tb_writer.add_scalar("evaluation/test", test_pref, epoch)
                for k, v in loss_dict.items():
                    tb_writer.add_scalar(f"training/{k}", v, epoch)

    best_val_epoch = np.argmin(np.array(valid_curve))
    if args.checkpoint_dir and args.enable_tb:
        tb_writer.close()
    print("Finished traning!")
    print(f"Best validation score: {valid_curve[best_val_epoch]}")
    print(f"Test score: {test_curve[best_val_epoch]}")


if __name__ == "__main__":
    main()
