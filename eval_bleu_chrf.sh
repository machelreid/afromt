#!/bin/bash
cat $1 | grep ^S | sort -V | cut -f2 > src
cat $1 | grep ^T | sort -V | cut -f2 > tgt
cat $1 | grep ^H | sort -V | cut -f3 > hyp
cat tgt | sacremoses detokenize > tgt.tmp 2> /dev/null
cat hyp | sacremoses detokenize > hyp.tmp 2> /dev/null
mv tgt.tmp tgt
mv hyp.tmp hyp

echo "chrF: $(echo $(sacrebleu -b -m chrf -w3 tgt < hyp) 100 | awk '{print $1 * $2}')"
echo "BLEU: $(sacrebleu -m bleu -w2 tgt < hyp)"

# remove these if you want to keep them
rm src
rm tgt
rm hyp
