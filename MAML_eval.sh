export CUDA_VISIBLE_DEVICES=0 #${X_SGE_CUDA_DEVICE}
export PATH="/home/dawna/gs534/Software/anaconda3/bin:$PATH"

mkdir -p LOG
mkdir -p models

python eval_maml.py \
    --data ./data_eval/fbk \
    --cuda \
    --label_scp ./lib/mlabs_test/ \
    --model LSTM \
    --window_size 200 \
    --overlap 100 \
    --nfea 41 \
    --nhid 256 \
    --nframeout 256 \
    --nsegout 128 \
    --lr 0.001 \
    --wdecay 1e-6 \
    --iterations 20000 \
    --clip 0.5 \
    --nlabels 10 \
    --npoints 5 \
    --ntasks 10 \
    --lr_a 0.06 \
    --nsteps 5 \
    --logfile LOG/mamleval.txt \
    --save models/maml_model.pt \
    --log-interval 20 \
    --shuffle \
