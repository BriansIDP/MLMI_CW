export CUDA_VISIBLE_DEVICES=${X_SGE_CUDA_DEVICE}
export PATH="/home/dawna/gs534/Software/anaconda3/bin:$PATH"

mkdir -p LOG
mkdir -p models

python train_maml.py \
    --data ./data/fbk \
    --cuda \
    --label_scp ./lib/mlabs/ \
    --model LSTM \
    --window_size 200 \
    --overlap 100 \
    --nfea 41 \
    --nhid 256 \
    --nframeout 256 \
    --nsegout 128 \
    --lr 0.001 \
    --wdecay 1e-3 \
    --iterations 3000 \
    --clip 0.5 \
    --nlabels 10 \
    --npoints 5 \
    --ntasks 12 \
    --lr_a 0.05 \
    --nsteps 2 \
    --logfile LOG/mamllog2.txt \
    --save models/maml_2step.pt \
    --log-interval 20 \
    --shuffle \
