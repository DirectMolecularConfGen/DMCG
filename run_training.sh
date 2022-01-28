#!/bin/bash
set -x
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

cd $SCRIPT_DIR/
if [ -z "$(pip list | grep confgen)" ]; then
    pip install -e . --user
fi

cuda=0
POSITIONAL=()
dist=false
prefix=vaeprior
port=29500
while [[ $# -gt 0 ]]; do
    key=$1
    case $key in
    -c | --cuda)
        cuda=$2
        shift 2
        ;;
    --dist)
        dist=true
        shift
        ;;
    --prefix)
        prefix=$2
        shift 2
        ;;
    --port)
        port=$2
        shift 2
        ;;
    *)
        POSITIONAL+=("$1")
        shift
        ;;
    esac
done
SUFFIX=$(echo ${POSITIONAL[*]} | sed -r 's/-//g' | sed -r 's/\s+/-/g')
SAVEDIR=/model/confgen/vae/$prefix
if [ -n "$SUFFIX" ]; then
    SAVEDIR=${SAVEDIR}-${SUFFIX}
fi
mkdir -p $SAVEDIR
if [ "$dist" == true ]; then
    cudaa=$(echo $cuda | sed -r 's/,//g')
    nproc=${#cudaa}
    CUDA_VISIBLE_DEVICES=$cuda python -m torch.distributed.launch --master_port $port --nproc_per_node=$nproc train.py \
        --checkpoint-dir $SAVEDIR ${POSITIONAL[@]} --lr-warmup --use-adamw --enable-tb | tee $SAVEDIR/training.log
else
    CUDA_VISIBLE_DEVICES=$cuda python train.py \
        --checkpoint-dir $SAVEDIR ${POSITIONAL[@]} --lr-warmup --use-adamw --enable-tb | tee $SAVEDIR/training.log
fi

# bash run_training.sh --dropout 0.1 --use-bn --lr-warmup --use-adamw --enable-tb --aux-loss 0.1 \
#     --num-layers 6 --lr 2e-4 --batch-size 128 --vae-beta-min 0.0001 --vae-beta-max 0.005 \
#     --reuse-prior --beta2 0.98 --data-split confgf --node-attn
