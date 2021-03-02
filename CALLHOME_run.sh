#!/bin/bash

INSTRUCTION=$1
METHOD=$2 # AHC or AHC+VB

exp_dir=$3 # output experiment directory
xvec_dir=$4 # output xvectors directory
WAV_DIR=$5 # wav files directory
FILE_LIST=$6 # txt list of files to process
LAB_DIR=$7 # lab files directory with VAD segments
#RTTM_DIR=$8 # reference rttm files directory
REF_RTTM=$8
QUEUE=${9:-none}

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"


if [[ $INSTRUCTION = "xvectors" ]]; then
    WEIGHTS_DIR=$DIR/VBx/models/ResNet101_8kHz/nnet
    if [ ! -f $WEIGHTS_DIR/raw_195.pth ]; then
        cat $WEIGHTS_DIR/raw_195.pth.zip.part* > $WEIGHTS_DIR/unsplit_raw_195.pth.zip
        unzip $WEIGHTS_DIR/unsplit_raw_195.pth.zip -d $WEIGHTS_DIR/
    fi

    WEIGHTS=$DIR/VBx/models/ResNet101_8kHz/nnet/raw_195.pth
    EXTRACT_SCRIPT=$DIR/VBx/extract.sh
    DEVICE=cpu

    mkdir -p $xvec_dir
    $EXTRACT_SCRIPT ResNet101 $WEIGHTS $WAV_DIR $LAB_DIR $FILE_LIST $xvec_dir $DEVICE

    if [ "$QUEUE" = "none" ]; then
        bash $xvec_dir/xv_task
    else
        nl=$(wc -l < $FILE_LIST)
        qsub -cwd -l num_proc=1,mem_free=4G,h_rt=400:00:00 -q $QUEUE -t 1:$nl -sync y -o $xvec_dir/extract.log -e $xvec_dir/extract.err $xvec_dir/uge_xv_task.sh
    fi
fi

BACKEND_DIR=$DIR/VBx/models/ResNet101_8kHz
if [[ $INSTRUCTION = "diarization" ]]; then
    TASKFILE=$exp_dir/diar_"$METHOD"_task
    UGE_TASKFILE=$exp_dir/diar_"$METHOD"_uge_task
    OUTFILE=$exp_dir/diar_"$METHOD"_out
    rm -f $TASKFILE $OUTFILE $UGE_TASKFILE
    mkdir -p $exp_dir/lists

    echo "#!/bin/bash" >> $UGE_TASKFILE
    echo ". /etc/profile.d/modules.sh" >> $UGE_TASKFILE
    echo "module load cuda11.0/blas/11.0.3 cuda11.0/toolkit/11.0.3 cudnn/8.0.2_cuda11.0"
    echo "conda deactivate" >> $UGE_TASKFILE
    echo "conda activate xvec" >> $UGE_TASKFILE
    printf "flist=(" >> $UGE_TASKFILE

    thr=-0.015
    tareng=0.3
    smooth=7.0
    lda_dim=128
    Fa=0.4
    Fb=17
    loopP=0.40
    OUT_DIR=$exp_dir/out_dir_"$METHOD"
    if [[ ! -d $OUT_DIR ]]; then
        mkdir -p $OUT_DIR
        while IFS= read -r line; do
            grep $line $FILE_LIST > $exp_dir/lists/$line".txt"
            echo "python $DIR/VBx/vbhmm.py --init $METHOD --out-rttm-dir $OUT_DIR/rttms --xvec-ark-file $xvec_dir/xvectors/$line.ark --segments-file $xvec_dir/segments/$line --plda-file $BACKEND_DIR/plda --xvec-transform $BACKEND_DIR/transform.h5 --threshold $thr --target-energy $tareng --init-smoothing $smooth --lda-dim $lda_dim --Fa $Fa --Fb $Fb --loopP $loopP" >> $TASKFILE
            printf "$line " >> $UGE_TASKFILE
        done < $FILE_LIST

        printf ")\n\n" >> $UGE_TASKFILE
        if [ "$QUEUE" = "none" ]; then
            bash $TASKFILE > $OUTFILE
        else
            # run it on the grid
            # TODO: add GPU support
            echo "file_uge=\${flist[\$((\${SGE_TASK_ID}-1))]}" >> $UGE_TASKFILE
            echo "python $DIR/VBx/vbhmm.py --init $METHOD --out-rttm-dir $OUT_DIR/rttms --xvec-ark-file $xvec_dir/xvectors/\${file_uge}.ark --segments-file $xvec_dir/segments/\${file_uge} --plda-file $BACKEND_DIR/plda --xvec-transform $BACKEND_DIR/transform.h5 --threshold $thr --target-energy $tareng --init-smoothing $smooth --lda-dim $lda_dim --Fa $Fa --Fb $Fb --loopP $loopP" >> $UGE_TASKFILE
            nl=$(wc -l < $FILE_LIST)
            qsub -cwd -l num_proc=1,mem_free=4G,h_rt=400:00:00 -N vbxhmm -q $QUEUE -t 1:$nl -sync y -o $OUT_DIR/vbhmm.log -e $OUT_DIR/vbhmm.err $UGE_TASKFILE
        fi
    fi
    ## Score
    cat $OUT_DIR/rttms/*.rttm > $OUT_DIR/sys.rttm
    #cat $RTTM_DIR/*.rttm > $OUT_DIR/ref.rttm
    cp $REF_RTTM $OUT_DIR/ref.rttm
    $DIR/dscore/score.py --collar 0.25 --ignore_overlaps -r $OUT_DIR/ref.rttm -s $OUT_DIR/sys.rttm > $OUT_DIR/result_forgiving
    $DIR/dscore/score.py --collar 0.25 -r $OUT_DIR/ref.rttm -s $OUT_DIR/sys.rttm > $OUT_DIR/result_fair
    $DIR/dscore/score.py --collar 0.0 -r $OUT_DIR/ref.rttm -s $OUT_DIR/sys.rttm > $OUT_DIR/result_full
fi
