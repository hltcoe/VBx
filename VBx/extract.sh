#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# @Authors: Federico Landini
# @Emails: landini@fit.vutbr.cz

# LOGIC
#  if model is a filename, weights is ignored and backend is assumed to be torch
#  if model is a string and weights is a file:
#       backend is assumed to be torch if weights ends in .pth
#       backend is assumed to be onnx if weights ends in .onnx

MODEL=$1
WEIGHTS=$2
WAV_DIR=$3
LAB_DIR=$4
FILE_LIST=$5
OUT_DIR=$6
DEVICE=$7

FEAT_EXTRACT_ENGINE=${8:-but}  # can be but or kaldi
KALDI_FBANK_CONF=${9:-none}
EMBED_DIM=${10:-256}  # xvector embedding dimension

# Error Checking for Input
# if the input is an ONNX model, then MODEL is a string defining the architecture
# and WEIGHTS is the file of weights
# if the input is a PyTorch model, MODEL is a path to the model
# and WEIGHTS will be ignored
if [ -f "$MODEL" ]; then
  backend=pytorch
  model_f=1
else
  model_f=0
  if [ -f "$WEIGHTS" ]; then
    weights_nopath=$(basename -- "$WEIGHTS")
    weights_ext="${weights_nopath##*.}"
    if [[ "$weights_ext" == "onnx" ]]; then
      backend=onnx
    elif [[ "$weights_ext" == "pth" || "$weights_ext" == "pt" ]]; then
      backend=pytorch
    else
      echo "Model weights file specified ("$WEIGHTS") is not recognized as ONNX or PyTorch!"
      exit -1
    fi
  else
    echo "Model specified is a string, but weights is not a .pth or .onnx - don't know how to proceed!"
    exit -1
  fi
fi
if [[ "$FEAT_EXTRACT_ENGINE" == "kaldi" ]]; then
  if [ ! -f "$KALDI_FBANK_CONF" ]; then
    echo "KALDI_FBANK_CONF must be specified when FEAT_EXTRACT_ENGINE=kaldi"
    exit -1
  fi
fi


DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

mkdir -pv $OUT_DIR

TASKFILE=$OUT_DIR/xv_task
UGE_TASKFILE=$OUT_DIR/uge_xv_task.sh
rm -f $TASKFILE
rm -f $UGE_TASKFILE

echo "#!/bin/bash" >> $UGE_TASKFILE
echo ". /etc/profile.d/modules.sh" >> $UGE_TASKFILE
echo "source deactivate" >> $UGE_TASKFILE
echo "source activate xvec" >> $UGE_TASKFILE
printf "flist=(" >> $UGE_TASKFILE


mkdir -p $OUT_DIR/lists $OUT_DIR/xvectors $OUT_DIR/segments
while IFS= read -r line; do
	mkdir -p "$(dirname $OUT_DIR/lists/$line)"
    grep $line $FILE_LIST > $OUT_DIR/lists/$line".txt"

    printf "$line " >> $UGE_TASKFILE

    OUT_ARK_FILE=$OUT_DIR/xvectors/$line.ark
    OUT_SEG_FILE=$OUT_DIR/segments/$line
    mkdir -p "$(dirname $OUT_ARK_FILE)"
    mkdir -p "$(dirname $OUT_SEG_FILE)"
    if [[ "$DEVICE" == "gpu" ]]; then
      if [[ "$backend" == "onnx" ]] || [[ "$backend" == "pytorch" && "$model_f" -eq 0 ]]; then
    	  echo "python $DIR/predict.py --seg-len 144 --seg-jump 24 --model $MODEL --weights $WEIGHTS --backend $backend --feat-extraction-engine $FEAT_EXTRACT_ENGINE --gpus=\$($DIR/free_gpu.sh) --ndim 64 --embed-dim $EMBED_DIM --in-file-list $OUT_DIR/lists/$line".txt" --in-lab-dir $LAB_DIR --in-wav-dir $WAV_DIR --out-ark-fn $OUT_ARK_FILE --out-seg-fn $OUT_SEG_FILE" >> $TASKFILE
    	else
  	    echo "python $DIR/predict.py --seg-len 144 --seg-jump 24 --model-file $MODEL --backend $backend --feat-extraction-engine $FEAT_EXTRACT_ENGINE --kaldi-fbank-conf $KALDI_FBANK_CONF --gpus=\$($DIR/free_gpu.sh) --ndim 64 --embed-dim $EMBED_DIM --in-file-list $OUT_DIR/lists/$line".txt" --in-lab-dir $LAB_DIR --in-wav-dir $WAV_DIR --out-ark-fn $OUT_ARK_FILE --out-seg-fn $OUT_SEG_FILE" >> $TASKFILE
    	fi
    else
      if [[ "$backend" == "onnx" ]] || [[ "$backend" == "pytorch" && "$model_f" -eq 0 ]]; then
    	  echo "python $DIR/predict.py --seg-len 144 --seg-jump 24 --model $MODEL --weights $WEIGHTS --backend $backend --feat-extraction-engine $FEAT_EXTRACT_ENGINE --ndim 64 --embed-dim $EMBED_DIM --in-file-list $OUT_DIR/lists/$line".txt" --in-lab-dir $LAB_DIR --in-wav-dir $WAV_DIR --out-ark-fn $OUT_ARK_FILE --out-seg-fn $OUT_SEG_FILE" >> $TASKFILE
    	else
    	  echo "python $DIR/predict.py --seg-len 144 --seg-jump 24 --model-file $MODEL --backend $backend --feat-extraction-engine $FEAT_EXTRACT_ENGINE --kaldi-fbank-conf $KALDI_FBANK_CONF --ndim 64 --embed-dim $EMBED_DIM --in-file-list $OUT_DIR/lists/$line".txt" --in-lab-dir $LAB_DIR --in-wav-dir $WAV_DIR --out-ark-fn $OUT_ARK_FILE --out-seg-fn $OUT_SEG_FILE" >> $TASKFILE
    	fi
    fi
done < $FILE_LIST

printf ")\n\n" >> $UGE_TASKFILE

# TODO: add GPU support
echo "file_uge=\${flist[\$((\${SGE_TASK_ID}-1))]}" >> $UGE_TASKFILE
echo "out_ark_file_uge=$OUT_DIR/xvectors/\${file_uge}.ark" >> $UGE_TASKFILE
echo "out_seg_file_uge=$OUT_DIR/segments/\${file_uge}" >> $UGE_TASKFILE
if [[ "$backend" == "onnx" ]] || [[ "$backend" == "pytorch" && "$model_f" -eq 0 ]]; then
  echo "python $DIR/predict.py --seg-len 144 --seg-jump 24 --model $MODEL --weights $WEIGHTS --backend $backend --feat-extraction-engine $FEAT_EXTRACT_ENGINE --ndim 64 --embed-dim $EMBED_DIM --in-file-list $OUT_DIR/lists/\${file_uge}".txt" --in-lab-dir $LAB_DIR --in-wav-dir $WAV_DIR --out-ark-fn \$out_ark_file_uge --out-seg-fn \$out_seg_file_uge" >> $UGE_TASKFILE
else
  echo "python $DIR/predict.py --seg-len 144 --seg-jump 24 --model-file $MODEL --backend $backend --feat-extraction-engine $FEAT_EXTRACT_ENGINE --kaldi-fbank-conf $KALDI_FBANK_CONF --ndim 64 --embed-dim $EMBED_DIM --in-file-list $OUT_DIR/lists/\${file_uge}".txt" --in-lab-dir $LAB_DIR --in-wav-dir $WAV_DIR --out-ark-fn \$out_ark_file_uge --out-seg-fn \$out_seg_file_uge" >> $UGE_TASKFILE
fi

