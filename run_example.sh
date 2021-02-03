#!/usr/bin/env bash

CDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

data_rootdir=/exp/kkarra/diarization/callhome
output_dir=/exp/kkarra/diarization/callhome/exp/

mkdir -p $output_dir
mkdir -p $output_dir/results

#for audio in $(ls example/audios/16k)
for audio in $(ls $data_rootdir/audio_wav)
do
      filename=$(echo "${audio}" | cut -f 1 -d '.')
      echo ${filename} > $output_dir/list.txt

      # run feature and x-vectors extraction
      python VBx/predict.py --in-file-list $output_dir/list.txt \
          --in-lab-dir $data_rootdir/sad_labels \
          --in-wav-dir $data_rootdir/audio_wav \
          --out-ark-fn $output_dir/${filename}.ark \
          --out-seg-fn $output_dir/${filename}.seg \
          --weights VBx/models/ResNet101_16kHz/nnet/final.onnx \
          --backend onnx
      #--in-lab-dir example/vad \
      #--in-wav-dir example/audios/16k \
 
      # run variational bayes on top of x-vectors
      python VBx/vbhmm.py --init AHC+VB \
          --out-rttm-dir $output_dir \
          --xvec-ark-file $output_dir/${filename}.ark \
          --segments-file $output_dir/${filename}.seg \
          --xvec-transform VBx/models/ResNet101_16kHz/transform.h5 \
          --plda-file VBx/models/ResNet101_16kHz/plda \
          --threshold -0.015 \
          --lda-dim 128 \
          --Fa 0.3 \
          --Fb 17 \
          --loopP 0.99

      # check if there is ground truth .rttm file
      if [ -f $data_rootdir/rttm_split/${filename}.rttm ]
      then
          # run dscore
          python dscore/score.py -r $data_rootdir/rttm_split/${filename}.rttm -s $output_dir/${filename}.rttm --collar 0.25 --ignore_overlaps > $output_dir/results/${filename}.out
      fi
done
