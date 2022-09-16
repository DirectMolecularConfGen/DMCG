# Direct Molecular Conformation Generation 
This repository contains the code for Direct Molecular Conformation Generation (DMCG).
## Dataset 
Download **rdkit_folder.tar.gz** from this [url](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JNGTDF).
```shell
tar -xvf rdkit_folder.tar.gz
```

## Requirements and Installation
* PyTorch
* Torch-Geometric

You can build a Docker image with the [Dockerfile](Dockerfile).
To install DMCG and develop it locally
```shell
pip install -e . 
```

## Train
The first time you run this code, you should specify the data path with `--base-path`, and the code will binarize data into binarized format.
```shell
# This is the example script for small-scale QM9.
bash run_training.sh --dropout 0.1 --use-bn --no-3drot  \
    --aux-loss 0.2 --num-layers 6 --lr 2e-4 --batch-size 128 --vae-beta-min 0.0001 --vae-beta-max 0.03 \
    --reuse-prior --node-attn --data-split confgf --pred-pos-residual \
    --dataset-name qm9 --remove-hs --shared-output  --base-path $yourdatapath

# Another example script for small-scale Drugs.
bash run_training.sh --dropout 0.1 --use-bn --no-3drot  \
    --aux-loss 0.2 --num-layers 6 --lr 2e-4 --batch-size 128 --vae-beta-min 0.0001 --vae-beta-max 0.05 \
    --reuse-prior --node-attn --data-split confgf --pred-pos-residual --grad-norm 10 \
    --dataset-name drugs --remove-hs --shared-output  --ang-lam 0 --bond-lam 0 --base-path $yourdatapath
```

We have provided the pretrained checkpoints and the corresponding logs on [GoogleDrive](https://drive.google.com/drive/folders/1PwXdDLZNSS8bc-kf3Xudd1Q6NySZojML?usp=sharing), and you can compare your configurations with our provided logs (specifically, the row started with "Namespace") to reproduce our results.
## Test
We have provided the inference scripts and the pretrained models on [GoogleDrive](https://drive.google.com/drive/folders/1PwXdDLZNSS8bc-kf3Xudd1Q6NySZojML?usp=sharing).
```shell
python evaluate.py --dropout 0.1 --use-bn --lr-warmup --use-adamw --train-subset \
    --num-layers 6 --eval-from  $yourcktpath --workers 20 --batch-size 128 \
    --reuse-prior --node-attn --data-split confgf --dataset-name qm9 --remove-hs \
    --shared-output --pred-pos-residual --sample-beta 1.2
```