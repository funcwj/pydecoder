#!/bin/bash

[ $# -ne 1 ] && echo "format error: $0: <wave-in>" && exit 1


online2-utt-nnet3-decoder --frames-per-chunk=20 --extra-left-context-initial=0  \
    --min-active=200 --max-active=7000 --beam=11.0 --lattice-beam=6.0 --acoustic-scale=0.1 \
    --config=exp/nnet3/tdnn_sp_online/conf/online.conf \
    exp/nnet3/tdnn_sp_online/final.mdl exp/sat1/graph/HCLG.fst $1 exp/sat1/graph/words.txt

#online2-wav-nnet3-latgen-faster --do-endpointing=false --frames-per-chunk=20 \
#    --extra-left-context-initial=0 --online=true \
#    --config=exp/nnet3/tdnn_sp_online/conf/online.conf \
#    --min-active=200 --max-active=7000 --beam=11.0 --lattice-beam=6.0 \
#    --acoustic-scale=0.1 --word-symbol-table=exp/sat1/graph/words.txt \
#    exp/nnet3/tdnn_sp_online/final.mdl \
#    exp/sat1/graph/HCLG.fst \
#   "ark:echo utt utt|" "scp:echo utt $1|" ark:/dev/null
