#!/usr/bin/env bash

CDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

mkdir -p exp

for audio in $(ls example/audios/16k)
do
      filename=$(echo "${audio}" | cut -f 1 -d '.')
      echo ${filename} > exp/list.txt

      # run feature and x-vectors extraction
      python VBx/predict.py --in-file-list exp/list.txt \
          --in-lab-dir example/vad \
          --in-wav-dir example/audios/16k \
          --out-ark-fn exp/${filename}.ark \
          --out-seg-fn exp/${filename}.seg \
          --model-file /expscratch/amccree/pytorch/v10_gauss_lnorm/adam_768_128_postvox/Test/models/checkpoint-epoch400.pth \
          --backend pytorch \
          --feat-extraction-engine kaldi \
          --kaldi-fbank-conf /expscratch/kkarra/train_egs/fbank_8k.conf
 
          # using COE xvectors 
          #--model-file /expscratch/amccree/pytorch/v10_gauss_lnorm/adam_768_128_postvox/Test/models/checkpoint-epoch400.pth \
          #--backend pytorch \
          #--feat-extraction-engine kaldi \
          #--kaldi-fbank-conf /expscratch/kkarra/train_egs/fbank_8k.conf
          # 
          # using BUT xvectors
          #--weights VBx/models/ResNet101_16kHz/nnet/final.onnx \
          #--backend onnx \
 
      # run variational bayes on top of x-vectors
      python VBx/vbhmm.py --init AHC+VB \
          --out-rttm-dir exp \
          --xvec-ark-file exp/${filename}.ark \
          --segments-file exp/${filename}.seg \
          --threshold -0.015 \
          --Fa 0.3 \
          --Fb 17 \
          --loopP 0.99 \
          --plda-file /expscratch/amccree/pytorch/v10_gauss_lnorm/adam_768_128_postvox/Test/models/checkpoint-epoch400.pth \
          --plda-format pytorch

          # using the COE plda
          #--plda-file /expscratch/amccree/pytorch/v10_gauss_lnorm/adam_768_128_postvox/Test/models/checkpoint-epoch400.pth \
          #--plda-format pytorch
          #
          # the BUT way
          #--lda-dim 128 \
          #--xvec-transform VBx/models/ResNet101_16kHz/transform.h5 \
          #--plda-file VBx/models/ResNet101_16kHz/plda
    
      # check if there is ground truth .rttm file
      if [ -f example/rttm/${filename}.rttm ]
      then
          # run dscore
          python dscore/score.py -r example/rttm/${filename}.rttm -s exp/${filename}.rttm --collar 0.25 --ignore_overlaps
      fi
done
