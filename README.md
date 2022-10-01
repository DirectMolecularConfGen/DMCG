# Direct Molecular Conformation Generation 
This repository contains the code for *Direct Molecular Conformation Generation (DMCG)*.

## Requirements and Installation

### Through Docker

We recommend building a Docker image with the [Dockerfile](Dockerfile).

After building and starting the docker, you can run

```shell
cd /workspace
git clone https://github.com/DirectMolecularConfGen/DMCG
cd DMCG
pip install -e .
```

You may possibly need to run `pip install setuptools==59.5.0` if you met problems with the `setuptools` module.

### Through conda venv
If you want to develop it locally using conda venv, please refer to Line 27 to Line 36 in [Dockerfile](Dockerfile) to build a virtual conda environment.

## Dataset 

### Small-scale GEOM-QM9 and GEOM-Drugs data
Download the **qm9_processed.7z** and **drugs_processed.7z** from this [url](https://drive.google.com/drive/folders/10dWaj5lyMY0VY4Zl0zDPCa69cuQUGb-6)


### Large-scale GEOM-QM9 and GEOM-Drugs data
Download **rdkit_folder.tar.gz** from this [url](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JNGTDF) and untar this file by `tar -xvf rdkit_folder.tar.gz`


## Training and inference


### Reproduce small-scale GEOM-Drugs
The first time you run this code, you should specify the data path with `--base-path`, and the code will binarize data into binarized format.
```shell
# Training. We place the unzipped data folder in /workspace/drugs_processed
DATA="/workspace/drugs_processed"
bash run_training.sh -c 0 --dropout 0.1 --use-bn --no-3drot \
    --aux-loss 0.2 --num-layers 6 --lr 2e-4 --batch-size 128 \
    --vae-beta-min 0.0001 --vae-beta-max 0.05 --reuse-prior \
    --node-attn --data-split confgf --pred-pos-residual --grad-norm 10 \
    --dataset-name drugs --remove-hs --shared-output \
    --ang-lam 0 --bond-lam 0 --base-path ${DATA}

# Inference. We recommend using checkpoint_94.pt
CKPT="/model/confgen/vae/vaeprior-dropout-0.1-usebn-no3drot-auxloss-0.2-numlayers-6-lr-2e4-batchsize-128-vaebetamin-0.0001-vaebetamax-0.05-reuseprior-nodeattn-datasplit-confgf-predposresidual-gradnorm-10-datasetname-drugs-removehs-sharedoutput-anglam-0-bondlam-0-basepath-/workspace/drugs_processed/checkpoint_94.pt"

python evaluate.py --dropout 0.1 --use-bn --lr-warmup --use-adamw --train-subset \
    --num-layers 6 --eval-from $CKPT --workers 20 --batch-size 128 \
    --reuse-prior --node-attn --data-split confgf --dataset-name drugs --remove-hs \
    --shared-output --pred-pos-residual --sample-beta 1.2
```

### Reproduce small-scale GEOM-QM9

```shell
# Training. We place the unzipped data folder in /workspace/qm9_processed
bash run_training.sh --dropout 0.1 --use-bn --no-3drot  \
    --aux-loss 0.2 --num-layers 6 --lr 2e-4 --batch-size 128 \
    --vae-beta-min 0.0001 --vae-beta-max 0.03 --reuse-prior \
    --node-attn --data-split confgf --pred-pos-residual \
    --dataset-name qm9 --remove-hs --shared-output  \
    --ang-lam 0.2 --bond-lam 0.1 --base-path $yourdatapath

# Inference. We recommend using checkpoint_94.pt
python evaluate.py --dropout 0.1 --use-bn --lr-warmup --use-adamw --train-subset \
    --num-layers 6 --eval-from $CKPT --workers 20 --batch-size 128 \
    --reuse-prior --node-attn --data-split confgf --dataset-name qm9 --remove-hs \
    --shared-output --pred-pos-residual --sample-beta 1.2
```

## Checkpoints and logs
We have provided the pretrained checkpoints and the corresponding logs on [GoogleDrive](https://drive.google.com/drive/folders/1PwXdDLZNSS8bc-kf3Xudd1Q6NySZojML?usp=sharing), and you can compare your configurations with our provided logs (specifically, the row started with "Namespace") to reproduce our results.
