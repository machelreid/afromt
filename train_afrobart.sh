langs="en,xh,af,zu,sw,bem,run,ln"
DATA=$1 # path to data-bin-afromt
OUTDIR=$2 # path to model logging/checkpointing dir
TGT_LANG=$3

mkdir -p $OUTDIR/log
PRETRAIN=afrobart.pt # change if need be


$HOME/miniconda3/envs/py3/bin/fairseq-train $DATA --fp16 \
--arch mbart_base --layernorm-embedding \ # same architecture as mbart
--task translation_from_xbart \
--source-lang en --target-lang $TGT_LANG \ # we don't use language tokens so technically this doesn't matter for performance -- just make sure it's in the list of languages at the top of the file and the language code of what you used during preprocessing
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
--lr-scheduler polynomial_decay --lr 3e-04 --stop-min-lr -1 --warmup-updates 5000 --total-num-update 50000 \ # change the warmup/total-num-update values depending on the size of your dataset!!
--eval-bleu --eval-bleu-remove-bpe \ 
--dropout 0.3 --weight-decay 0.01 --attention-dropout 0.1 \
--max-tokens 4096 --update-freq 2 \ #Change this depending on the size of your GPU
--validate-interval=10 --save-interval=10 --save-interval-updates 1000 --keep-interval-updates 2 --no-epoch-checkpoints --validate-after-updates 2500 \ # Change these depending on the size of your dataset!!
--seed 666 --log-format simple --log-interval 20 \
--langs $langs --save-dir $OUTDIR/checkpoints \
--skip-invalid-size-inputs-valid-test --tensorboard-logdir $OUTDIR/tensorboard \
--decoder-embed-dim=512 \
--decoder-ffn-embed-dim=2048 \
--restore-file $PRETRAIN --save-dir $OUTDIR/checkpoints \
--reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \
--encoder-embed-dim=512 \
--encoder-ffn-embed-dim=2048 \
--encoder-layers=6 \
--encoder-attention-heads=8 \
--decoder-layers=6 --decoder-attention-heads=8 \
--ddp-backend=no_c10d | tee -a $OUTDIR/train.log
