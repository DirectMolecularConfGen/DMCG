import math
import copy
from rdkit.Chem.rdmolops import RemoveHs
from rdkit.Chem import rdMolAlign as MA
import torch
import os
import numpy as np
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from tqdm import tqdm


class WarmCosine:
    def __init__(self, warmup=4e3, tmax=1e5, eta_min=5e-4):
        if warmup is None:
            self.warmup = 0
        else:
            warmup_step = int(warmup)
            assert warmup_step > 0
            self.warmup = warmup_step
            self.lr_step = (1 - eta_min) / warmup_step
        self.tmax = int(tmax)
        self.eta_min = eta_min

    def step(self, step):
        if step >= self.warmup:
            return (
                self.eta_min
                + (1 - self.eta_min)
                * (1 + math.cos(math.pi * (step - self.warmup) / self.tmax))
                / 2
            )

        else:
            return self.eta_min + self.lr_step * step


def set_rdmol_positions(rdkit_mol, pos):
    assert rdkit_mol.GetConformer(0).GetPositions().shape[0] == pos.shape[0]
    mol = copy.deepcopy(rdkit_mol)
    for i in range(pos.shape[0]):
        mol.GetConformer(0).SetAtomPosition(i, pos[i].tolist())
    return mol


def get_best_rmsd(gen_mol, ref_mol):
    gen_mol = RemoveHs(gen_mol)
    ref_mol = RemoveHs(ref_mol)
    rmsd = MA.GetBestRMS(gen_mol, ref_mol)
    return rmsd


def get_random_rotation_3d(pos):
    random_quaternions = torch.randn(4).to(pos)
    random_quaternions = random_quaternions / random_quaternions.norm(dim=-1, keepdim=True)
    return torch.einsum("kj,ij->ki", pos, quaternion_to_rotation_matrix(random_quaternions))


def quaternion_to_rotation_matrix(quaternion):
    q0 = quaternion[0]
    q1 = quaternion[1]
    q2 = quaternion[2]
    q3 = quaternion[3]
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
    return torch.stack([r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=-1).reshape(3, 3)


class Cosinebeta:
    def __init__(self, args):
        if args.vae_beta_max is None or args.vae_beta_min is None:
            assert args.vae_beta is not None

        self.vae_beta_max = args.vae_beta_max
        self.vae_beta_min = args.vae_beta_min
        self.args = args
        self.T = 2 * args.epochs

    def step(self, epoch):
        if self.vae_beta_max is None or self.vae_beta_min is None:
            return
        self.args.vae_beta = self.vae_beta_min - 1 / 2 * (self.vae_beta_max - self.vae_beta_min) * (
            math.cos(2 * math.pi / self.T * epoch) - 1
        )


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True
    torch.cuda.set_device(args.local_rank)
    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {} local rank {}): {}".format(
            args.rank, args.local_rank, "env://"
        ),
        flush=True,
    )
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method="env://", world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def binarize(x):
    return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))


def evaluate_distance(gen_mols, ref_mols):
    gen_mols = gen_mols[:1000]
    pos_ref = [mol.GetConformer(0).GetPositions() for mol in ref_mols]
    pos_gen = [mol.GetConformer(0).GetPositions() for mol in gen_mols]

    pos_ref = torch.as_tensor(np.stack(pos_ref), dtype=torch.float32)
    pos_gen = torch.as_tensor(np.stack(pos_gen), dtype=torch.float32)

    edge_list = []
    mol = gen_mols[0]
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        if any(
            [mol.GetAtomWithIdx(i).GetAtomicNum() == 1, mol.GetAtomWithIdx(j).GetAtomicNum() == 1]
        ):
            continue
        edge_list.append((i, j))
        edge_list.append((j, i))
    edge_index = np.array(edge_list, dtype=np.int64).T
    edge_index = torch.as_tensor(edge_index)

    adj = to_dense_adj(edge_index).squeeze(0)
    adj_mats = [
        torch.eye(adj.size(0), dtype=torch.long),
        binarize(adj + torch.eye(adj.size(0), dtype=torch.long)),
    ]

    for i in range(2, 4):
        adj_mats.append(binarize(adj_mats[i - 1] @ adj_mats[1]))
    edge_index = dense_to_sparse(adj_mats[-1] - adj_mats[0])[0]

    ref_lengths = (pos_ref[:, edge_index[0]] - pos_ref[:, edge_index[1]]).norm(dim=-1)
    gen_lengths = (pos_gen[:, edge_index[0]] - pos_gen[:, edge_index[1]]).norm(dim=-1)

    stats_single = []
    first = True
    for i, (row, col) in enumerate(tqdm(edge_index.t())):
        if row >= col:
            continue
        gen_l = gen_lengths[:, i]
        ref_l = ref_lengths[:, i]
        if first:
            print(gen_l.size(), ref_l.size())
            first = False
        mmd = compute_mmd(gen_l.view(-1, 1).cuda(), ref_l.view(-1, 1).cuda()).item()
        stats_single.append({"mmd": mmd})

    stats_pair = []
    for i, (row_i, col_i) in enumerate(tqdm(edge_index.t())):
        if row_i >= col_i:
            continue
        for j, (row_j, col_j) in enumerate(edge_index.t()):
            if (row_i >= row_j) or (row_j >= col_j):
                continue
            gen_l = gen_lengths[:, (i, j)]
            ref_l = ref_lengths[:, (i, j)]

            mmd = compute_mmd(gen_l.cuda(), ref_l.cuda()).item()

            stats_pair.append({"mmd": mmd})

    edge_filter = edge_index[0] < edge_index[1]
    gen_l = gen_lengths[:, edge_filter]
    ref_l = ref_lengths[:, edge_filter]
    mmd = compute_mmd(gen_l.cuda(), ref_l.cuda()).item()

    stats_all = {"mmd": mmd}
    return stats_single, stats_pair, stats_all


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    Params:
	    source: n * len(x)
	    target: m * len(y)
	Return:
		sum(kernel_val): Sum of various kernel matrices
    """
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))

    L2_distance = ((total0 - total1) ** 2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)

    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

    return sum(kernel_val)  # /len(kernel_val)


def compute_mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    Params:
	    source: (N, D)
	    target: (M, D)
	Return:
		loss: MMD loss
    """
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(
        source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma
    )

    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX) + torch.mean(YY) - torch.mean(XY) - torch.mean(YX)

    return loss


def clip_norm(vec, limit, p=2):
    norm = torch.norm(vec, dim=-1, p=2, keepdim=True)
    denom = torch.where(norm > limit, limit / norm, torch.ones_like(norm))
    return vec * denom

