# AfroMT

Code for the EMNLP 2021 Paper [AfroMT: Pretraining Strategies and Reproducible Benchmarks for Translation of 8 African Languages](https://arxiv.org/abs/2109.04715). 

```bibtex
@inproceedings{reid21afromt,
    title = {Afro{MT}: Pretraining Strategies and Reproducible Benchmarks for Translation of 8 African Languages},
    author = {Machel Reid and Junjie Hu and Graham Neubig and Yutaka Matsuo},
    booktitle = {Conference on Empirical Methods in Natural Language Processing (EMNLP)},
    address = {Punta Cana, Dominican Republic},
    month = {November},
    url = {https://arxiv.org/abs/2109.04715},
    year = {2021}
}
```
## Clone the repo

```bash
git clone git@github.com:machelreid/afromt
```
## Data

Please use `gdown` to download the data
```bash
cd afromt
gdown --id 1Qj3IXQ9kusaeHtGYRVPzMnGm_dQvvfJP
tar -xf afromt.tar.xz
```
## Installation
Run the following commands
```bash
cd training
# install apex for fp16 for faster training
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ../
# install fairseq
pip install -e .
cd ../
```

## Model usage
Download the model as follows:
```bash
gdown --id 1A9kbWHnMrjFwgq9x8rpnU4aATx_Y4b3b
tar -xf afrobart.tar.xz # and you will get afrobart.pt which is the model file
```

### Preprocessing
First be sure to install [sentencepiece](https://github.com/google/sentencepiece) 
```bash
TGT_LANG=#whatever target language you choose
cd afromt # the data folder
cd en-$TGT_LANG
bash ../../preprocess.sh $TGT_LANG ../../
cd ../../
```
to preprocess \$TGT_LANG (e.g. `zu` for Zulu, or `xh` for Xhosa). This command will produce a `data-bin-afromt` folder in the folder for the en-\$TGT_LANG language pair.

### Training
More details within the script itself if you want to tweak training arguments!
```bash
bash train_afrobart.sh afromt/en-$TGT_LANG/data-bin-afromt afromt/en-$TGT_LANG/model_output $TGT_LANG
```
and you should see your model training!

### Evaluation
Once training is over, you can generate from the model as follows:
```bash
fairseq-generate afromt/en-$TGT_LANG/data-bin-afromt --gen-subset test --path afromt/en-$TGT_LANG/model_outputs/checkpoints/checkpoint_best.pt  --beam 5 --batch-size 300 --remove-bpe sentencepiece --truncate-source --task translation_from_xbart -s en -t $TGT_LANG --lenpen $LENPEN > generation.log 
```
(where $TGT_LANG and $LENPEN are variables controlling target language and length penalty respectively)

and generate BLEU scores as follows:
```bash
bash eval_bleu_chrf.sh generation.log
```


## Other tidbits
- The bilingual dictionaries we extracted are in the `dictionaries/` folder (note that these are automatically extracted with a word aligned so they are not gold standard)
