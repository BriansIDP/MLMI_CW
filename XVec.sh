export CUDA_VISIBLE_DEVICES=${X_SGE_CUDA_DEVICE}
export PATH="/home/dawna/gs534/Software/anaconda3/bin:$PATH"

mkdir -p LOG
mkdir -p models

python train.py \
    --data ./data/fbk \
    --cuda \
    --label_scp ./lib/mlabs/ \
    --model TDNN \
    --window_size 200 \
    --overlap 100 \
    --bsize 200 \
    --nfea 41 \
    --nhid 512 \
    --nframeout 1500 \
    --nsegout 512 \
    --lr 1.0 \
    --wdecay 1e-6 \
    --epochs 20 \
    --logfile LOG/log.txt \
    --save models/xvec_model.pt \
    --log-interval 100 \
    --shuffle \
