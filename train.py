import os
import argparse
import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import random
from confgen.e2c.dataset import PygGeomDataset
from confgen.model.gnn import GNN
from torch.optim.lr_scheduler import LambdaLR
from confgen.utils.utils import (
    Cosinebeta,
    WarmCosine,
    set_rdmol_positions,
    get_best_rmsd,
    init_distributed_mode,
)
import io
import json
from collections import defaultdict
from torch.utils.data import DistributedSampler


def train(model, device, loader, optimizer, scheduler, args):
    model.train()
    loss_accum_dict = defaultdict(float)
    pbar = tqdm(loader, desc="Iteration", disable=args.disable_tqdm)
    for step, batch in enumerate(pbar):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            atom_pred_list, extra_output = model(batch)
            optimizer.zero_grad()

            if args.distributed:
                loss, loss_dict = model.module.compute_loss(
                    atom_pred_list, extra_output, batch, args
                )
            else:
                loss, loss_dict = model.compute_loss(atom_pred_list, extra_output, batch, args)

            loss.backward()
            if args.grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            optimizer.step()
            scheduler.step()

            for k, v in loss_dict.items():
                loss_accum_dict[k] += v.detach().item()

            if step % args.log_interval == 0:
                description = f"Iteration loss: {loss_accum_dict['loss'] / (step + 1):6.4f}"
                description += f" lr: {scheduler.get_last_lr()[0]:.5e}"
                description += f" vae_beta: {args.vae_beta:6.4f}"
                # for k in loss_accum_dict.keys():
                #     description += f" {k}: {loss_accum_dict[k]/(step+1):6.4f}"

                pbar.set_description(description)

    for k in loss_accum_dict.keys():
        loss_accum_dict[k] /= step + 1
    return loss_accum_dict


def evaluate(model, device, loader, args):
    model.eval()
    mol_labels = []
    mol_preds = []
    for batch in tqdm(loader, desc="Iteration", disable=args.disable_tqdm):
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
            rmsd_list.append(get_best_rmsd(gen_mol, ref_mol))
        except Exception as e:
            continue

    return np.mean(rmsd_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--global-reducer", type=str, default="sum")
    parser.add_argument("--node-reducer", type=str, default="sum")
    parser.add_argument("--graph-pooling", type=str, default="sum")
    parser.add_argument("--dropedge-rate", type=float, default=0.1)
    parser.add_argument("--dropnode-rate", type=float, default=0.1)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--decoder-layers", type=int, default=None)
    parser.add_argument("--latent-size", type=int, default=256)
    parser.add_argument("--mlp-hidden-size", type=int, default=1024)
    parser.add_argument("--mlp_layers", type=int, default=2)
    parser.add_argument("--use-layer-norm", action="store_true", default=False)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4)
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

    parser.add_argument("--base-path", type=str, default="/data/code/ConfGF/dataset/")
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
    parser.add_argument("--extend-edge", action="store_true", default=False)
    parser.add_argument(
        "--data-split", type=str, choices=["cgcf", "default", "confgf"], default="default"
    )
    parser.add_argument("--reuse-prior", action="store_true", default=False)
    parser.add_argument("--cycle", type=int, default=1)

    parser.add_argument("--vae-beta", type=float, default=1.0)
    parser.add_argument("--vae-beta-max", type=float, default=None)
    parser.add_argument("--vae-beta-min", type=float, default=None)
    parser.add_argument("--pred-pos-residual", action="store_true", default=False)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--node-attn", action="store_true", default=False)
    parser.add_argument("--global-attn", action="store_true", default=False)
    parser.add_argument("--shared-decoder", action="store_true", default=False)
    parser.add_argument("--shared-output", action="store_true", default=False)
    parser.add_argument("--clamp-dist", type=float, default=None)
    parser.add_argument("--use-global", action="store_true", default=False)
    parser.add_argument("--sg-pos", action="store_true", default=False)
    parser.add_argument("--remove-hs", action="store_true", default=False)
    parser.add_argument("--grad-norm", type=float, default=None)
    parser.add_argument("--use-ss", action="store_true", default=False)
    parser.add_argument("--rand-aug", action="store_true", default=False)
    parser.add_argument("--not-origin", action="store_true", default=False)
    parser.add_argument("--ang-lam", type=float, default=0.)
    parser.add_argument("--bond-lam", type=float, default=0.)
    parser.add_argument("--no-3drot", action="store_true", default=False)

    args = parser.parse_args()

    init_distributed_mode(args)
    print(args)

    CosineBeta = Cosinebeta(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device)

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
    dataset_train = (
        dataset[split_idx["train"]]
        if not args.train_subset
        else dataset[split_idx["train"]][:102400]
    )

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True
    )

    train_loader = DataLoader(
        dataset_train, batch_sampler=batch_sampler_train, num_workers=args.num_workers,
    )
    train_loader_dev = DataLoader(
        dataset[split_idx["train"]][:102400],
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
    )
    valid_loader = DataLoader(
        dataset[split_idx["valid"]],
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        dataset[split_idx["test"]],
        batch_size=args.batch_size * 2,
        shuffle=False,
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
        "use_global": args.use_global,
        "sg_pos": args.sg_pos,
        "shared_output": args.shared_output,
        "use_ss": args.use_ss,
        "rand_aug": args.rand_aug,
        "no_3drot": args.no_3drot,
        "not_origin": args.not_origin,
    }
    model = GNN(**shared_params).to(device)
    model_without_ddp = model
    args.disable_tqdm = False
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
        model_without_ddp = model.module

        args.checkpoint_dir = "" if args.rank != 0 else args.checkpoint_dir
        args.enable_tb = False if args.rank != 0 else args.enable_tb
        args.disable_tqdm = args.rank != 0

    if args.eval_from is not None:
        assert os.path.exists(args.eval_from)
        checkpoint = torch.load(args.eval_from, map_location=device)["model_state_dict"]
        model_without_ddp.load_state_dict(checkpoint)

    num_params = sum(p.numel() for p in model_without_ddp.parameters())
    print(f"#Params: {num_params}")
    if args.use_adamw:
        optimizer = optim.AdamW(
            model_without_ddp.parameters(),
            lr=args.lr,
            betas=(0.9, args.beta2),
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = optim.Adam(
            model_without_ddp.parameters(),
            lr=args.lr,
            betas=(0.9, args.beta2),
            weight_decay=args.weight_decay,
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

    for epoch in range(1, args.epochs + 1):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        CosineBeta.step(epoch - 1)
        print("=====Epoch {}".format(epoch))
        print("Training...")
        loss_dict = train(model, device, train_loader, optimizer, scheduler, args)
        print("Evaluating...")
        train_pref = evaluate(model, device, train_loader_dev, args)
        valid_pref = evaluate(model, device, valid_loader, args)
        test_pref = evaluate(model, device, test_loader, args)

        if args.checkpoint_dir:
            print(f"Setting {os.path.basename(os.path.normpath(args.checkpoint_dir))}...")
        print(f"Train: {train_pref} Validation: {valid_pref} Test: {test_pref}")

        train_curve.append(train_pref)
        valid_curve.append(valid_pref)
        test_curve.append(test_pref)
        if args.checkpoint_dir:
            logs = {"Train": train_pref, "Valid": valid_pref, "Test": test_pref}
            with io.open(
                os.path.join(args.checkpoint_dir, "log.txt"), "a", encoding="utf8", newline="\n"
            ) as tgt:
                print(json.dumps(logs), file=tgt)

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model_without_ddp.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "args": args,
            }
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, f"checkpoint_{epoch}.pt"))
            if args.enable_tb:
                tb_writer.add_scalar("evaluation/train", train_pref, epoch)
                tb_writer.add_scalar("evaluation/valid", valid_pref, epoch)
                tb_writer.add_scalar("evaluation/test", test_pref, epoch)
                for k, v in loss_dict.items():
                    tb_writer.add_scalar(f"training/{k}", v, epoch)

    best_val_epoch = np.argmin(np.array(valid_curve))
    if args.checkpoint_dir and args.enable_tb:
        tb_writer.close()
    if args.distributed:
        torch.distributed.destroy_process_group()
    print("Finished traning!")
    print(f"Best validation epoch: {best_val_epoch+1}")
    print(f"Best validation score: {valid_curve[best_val_epoch]}")
    print(f"Test score: {test_curve[best_val_epoch]}")


if __name__ == "__main__":
    main()
