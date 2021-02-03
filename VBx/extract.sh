#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# @Authors: Federico Landini
# @Emails: landini@fit.vutbr.cz

MODEL=$1
WEIGHTS=$2
WAV_DIR=$3
LAB_DIR=$4
FILE_LIST=$5
OUT_DIR=$6
DEVICE=$7

EMBED_DIM=256
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
    	echo "python $DIR/predict.py --seg-len 144 --seg-jump 24 --model $MODEL --weights $WEIGHTS --gpus=\$($DIR/free_gpu.sh) $MDL_WEIGHTS --ndim 64 --embed-dim $EMBED_DIM --in-file-list $OUT_DIR/lists/$line".txt" --in-lab-dir $LAB_DIR --in-wav-dir $WAV_DIR --out-ark-fn $OUT_ARK_FILE --out-seg-fn $OUT_SEG_FILE" >> $TASKFILE
    else
    	echo "python $DIR/predict.py --seg-len 144 --seg-jump 24 --model $MODEL --weights $WEIGHTS --gpus= $MDL_WEIGHTS --ndim 64 --embed-dim $EMBED_DIM --in-file-list $OUT_DIR/lists/$line".txt" --in-lab-dir $LAB_DIR --in-wav-dir $WAV_DIR --out-ark-fn $OUT_ARK_FILE --out-seg-fn $OUT_SEG_FILE" >> $TASKFILE
    fi
done < $FILE_LIST

printf ")\n\n" >> $UGE_TASKFILE

# TODO: add GPU support
echo "file_uge=\${flist[\$((\${SGE_TASK_ID}-1))]}" >> $UGE_TASKFILE
echo "out_ark_file_uge=$OUT_DIR/xvectors/\${file_uge}.ark" >> $UGE_TASKFILE
echo "out_seg_file_uge=$OUT_DIR/segments/\${file_uge}" >> $UGE_TASKFILE
echo "python $DIR/predict.py --seg-len 144 --seg-jump 24 --model $MODEL --weights $WEIGHTS --gpus= $MDL_WEIGHTS --ndim 64 --embed-dim $EMBED_DIM --in-file-list $OUT_DIR/lists/\${file_uge}".txt" --in-lab-dir $LAB_DIR --in-wav-dir $WAV_DIR --out-ark-fn \$out_ark_file_uge --out-seg-fn \$out_seg_file_uge" >> $UGE_TASKFILE

