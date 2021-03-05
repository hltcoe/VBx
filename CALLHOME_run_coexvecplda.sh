#!/bin/bash

INSTRUCTION=$1
METHOD=$2        # AHC or AHC+VB or AHC+GMM

exp_dir=$3       # output experiment directory
xvec_dir_base=$4 # output xvectors directory
WAV_DIR=$5       # wav files directory
FILE_LIST=$6     # txt list of files to process
LAB_DIR=$7       # lab files directory with VAD segments
#RTTM_DIR=$8 # reference rttm files directory
REF_RTTM=$8
NUM_PASS=${9:-1}
QUEUE=${10:-none}

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# define models and configurations for each pass
#XVEC_PLDA_MODEL1=/expscratch/amccree/pytorch/v10_gauss_lnorm/adam_768_128_postvox/Test/models/checkpoint-epoch400.pth
#XVEC_PLDA_MODEL2=/expscratch/amccree/pytorch/v10_gauss_lnorm/adam_768_128_postvox/Test/models/checkpoint-epoch400.pth
#PASS1_SEG_JUMP=24
#PASS1_SEG_LEN=144
#PASS2_SEG_JUMP=24
#PASS2_SEG_LEN=144
#####
# configuration comes from /expscratch/amccree/pytorch/Diarize/v1/Test_clean/run_diar_2pass.py
XVEC_PLDA_MODEL1="/expscratch/amccree/pytorch/v10_gauss_lnorm/adam_768_128_postvox/Test_sgd/Updated/Test_freeze_2s_dc_vb/Test_1gpu/models/checkpoint-latest.pth"
XVEC_PLDA_MODEL2=$XVEC_PLDA_MODEL1
PASS1_SEG_JUMP=200  # corresponds to ovlp=0
PASS1_SEG_LEN=200   # corresponds to 2 sec
PASS2_SEG_JUMP=12   # corresponds to ovlp=0.5
PASS2_SEG_LEN=25    # corresponds to .25 sec
###

FEAT_EXTRACT_ENGINE=kaldi
KALDI_FBANK_CONF=/expscratch/kkarra/train_egs/fbank_8k.conf
EMBED_DIM=128

if (("$NUM_PASS" < 1)); then
  echo "NUM_PASS must be >= 1"
  exit 1
fi
passes=($(seq 1 1 $NUM_PASS))

# verify models/seg_jump/seg_len for each pass is defined
for pass in ${passes[@]}; do
  # model, seg_jump, and seg_len are dynamic bash variables.
  # See: https://stackoverflow.com/a/65021258/1057098
  model="XVEC_PLDA_MODEL$pass"
  seg_jump="PASS${pass}_SEG_JUMP"
  seg_len="PASS${pass}_SEG_LEN"
  if [ -z ${!model} ]; then
    echo "$model is undefined!"
    exit 1
  fi
  if [ -z ${!seg_jump} ]; then
    echo "$seg_jump is undefined!"
    exit 1
  fi
  if [ -z ${!seg_len} ]; then
    echo "$seg_len is undefined!"
    exit 1
  fi
done

if [[ $INSTRUCTION == "xvectors" ]]; then

  EXTRACT_SCRIPT=$DIR/VBx/extract.sh
  DEVICE=cpu

  for pass in "${passes[@]}"; do
    xvec_dir="$xvec_dir_base"_"$pass"
    mkdir -p $xvec_dir
    ##################################
    # model, seg_jump, and seg_len are dynamic bash variables.
    # See: https://stackoverflow.com/a/65021258/1057098
    model="XVEC_PLDA_MODEL$pass"
    seg_jump="PASS${pass}_SEG_JUMP"
    seg_len="PASS${pass}_SEG_LEN"
    ##################################
    $EXTRACT_SCRIPT ${!model} None $WAV_DIR $LAB_DIR $FILE_LIST $xvec_dir $DEVICE \
      $FEAT_EXTRACT_ENGINE $KALDI_FBANK_CONF $EMBED_DIM \
      ${!seg_jump} ${!seg_len}

    if [ "$QUEUE" = "none" ]; then
      bash $xvec_dir/xv_task
    else
      nl=$(wc -l <$FILE_LIST)
      qsub -cwd -l num_proc=1,mem_free=4G,h_rt=400:00:00 -q $QUEUE -t 1:$nl -sync y -o $xvec_dir/extract.log -e $xvec_dir/extract.err $xvec_dir/uge_xv_task.sh
    fi
  done
fi

BACKEND_DIR=$DIR/VBx/models/ResNet101_8kHz
if [[ $INSTRUCTION == "diarization" ]]; then
  TASKFILE=$exp_dir/diar_"$METHOD"_task
  UGE_TASKFILE=$exp_dir/diar_"$METHOD"_uge_task
  OUTFILE=$exp_dir/diar_"$METHOD"_out
  rm -f $TASKFILE $OUTFILE $UGE_TASKFILE
  mkdir -p $exp_dir/lists

  echo "#!/bin/bash" >>$UGE_TASKFILE
  echo ". /etc/profile.d/modules.sh" >>$UGE_TASKFILE
  echo "module load cuda11.0/blas/11.0.3 cuda11.0/toolkit/11.0.3 cudnn/8.0.2_cuda11.0"
  echo "source deactivate" >>$UGE_TASKFILE
  echo "source activate xvec" >>$UGE_TASKFILE
  printf "flist=(" >>$UGE_TASKFILE

  thr=-0.015
  tareng=0.3
  smooth=7.0
  Fa=0.4
  Fb=17
  loopP=0.40
  OUT_DIR=$exp_dir/out_dir_"$METHOD"
  if [[ ! -d $OUT_DIR ]]; then
    mkdir -p $OUT_DIR
    while IFS= read -r line; do
      grep $line $FILE_LIST >$exp_dir/lists/$line".txt"
      #echo "python $DIR/VBx/vbhmm.py --init $METHOD --out-rttm-dir $OUT_DIR/rttms --xvec-ark-file $xvec_dir/xvectors/$line.ark --segments-file $xvec_dir/segments/$line --plda-file $XVEC_PLDA_MODEL --plda-format pytorch --threshold $thr --target-energy $tareng --init-smoothing $smooth --Fa $Fa --Fb $Fb --loopP $loopP" >>$TASKFILE
      cmd_str="python $DIR/VBx/vbhmm.py --init $METHOD --out-rttm-dir $OUT_DIR/rttms --xvec-ark-file"
      xvec_inputarg=""
      for pass in "${passes[@]}"; do
        xvec_dir="$xvec_dir_base"_"$pass"
        xvec_inputarg="$xvec_inputarg ${xvec_dir}/xvectors/${line}.ark"
      done
      cmd_str="$cmd_str $xvec_inputarg  --segments-file"
      segment_inputarg=""
      for pass in "${passes[@]}"; do
        xvec_dir="$xvec_dir_base"_"$pass"
        segment_inputarg="$segment_inputarg ${xvec_dir}/segments/${line}"
      done
      cmd_str="$cmd_str $segment_inputarg  --plda-file"
      plda_inputarg=""
      for pass in "${passes[@]}"; do
        plda_pass_model="XVEC_PLDA_MODEL$pass"
        plda_inputarg="$plda_inputarg ${!plda_pass_model}"
      done
      cmd_str="$cmd_str $plda_inputarg --plda-format pytorch --threshold $thr --target-energy $tareng --init-smoothing $smooth --Fa $Fa --Fb $Fb --loopP $loopP"
      echo $cmd_str >> $TASKFILE

      printf "$line " >>$UGE_TASKFILE
    done <$FILE_LIST

    printf ")\n\n" >>$UGE_TASKFILE
    if [ "$QUEUE" = "none" ]; then
      bash $TASKFILE >$OUTFILE
    else
      # run it on the grid
      # TODO: add GPU support

      xvec_inputarg=""
      segment_inputarg=""
      plda_inputarg=""
      for pass in "${passes[@]}"; do
        xvec_dir="$xvec_dir_base"_"$pass"
        xvec_inputarg="$xvec_inputarg ${xvec_dir}/xvectors/\${file_uge}.ark"
        segment_inputarg="$segment_inputarg ${xvec_dir}/segments/\${file_uge}"
        plda_model="XVEC_PLDA_MODEL$pass"
        plda_inputarg="$plda_inputarg ${!plda_model}"
      done

      echo "file_uge=\${flist[\$((\${SGE_TASK_ID}-1))]}" >>$UGE_TASKFILE
      #echo "python $DIR/VBx/vbhmm.py --init $METHOD --out-rttm-dir $OUT_DIR/rttms --xvec-ark-file $xvec_dir/xvectors/\${file_uge}.ark --segments-file $xvec_dir/segments/\${file_uge} --plda-file $XVEC_PLDA_MODEL --plda-format pytorch --threshold $thr --target-energy $tareng --init-smoothing $smooth --Fa $Fa --Fb $Fb --loopP $loopP" >>$UGE_TASKFILE
      cmd_str="python $DIR/VBx/vbhmm.py --init $METHOD --out-rttm-dir $OUT_DIR/rttms --xvec-ark-file $xvec_inputarg --segments-file $segment_inputarg --plda-file $plda_inputarg --plda-format pytorch --threshold $thr --target-energy $tareng --init-smoothing $smooth --Fa $Fa --Fb $Fb --loopP $loopP"

      echo $cmd_str >> $UGE_TASKFILE
      nl=$(wc -l <$FILE_LIST)
      qsub -cwd -l num_proc=1,mem_free=4G,h_rt=400:00:00 -N vbxhmm -q $QUEUE -t 1:$nl -sync y -o $OUT_DIR/vbhmm.log -e $OUT_DIR/vbhmm.err $UGE_TASKFILE
    fi
  fi

  ## Score
  cat $OUT_DIR/rttms/*.rttm >$OUT_DIR/sys.rttm
  #cat $RTTM_DIR/*.rttm > $OUT_DIR/ref.rttm
  cp $REF_RTTM $OUT_DIR/ref.rttm
  $DIR/dscore/score.py --collar 0.25 --ignore_overlaps -r $OUT_DIR/ref.rttm -s $OUT_DIR/sys.rttm >$OUT_DIR/result_forgiving
  $DIR/dscore/score.py --collar 0.25 -r $OUT_DIR/ref.rttm -s $OUT_DIR/sys.rttm >$OUT_DIR/result_fair
  $DIR/dscore/score.py --collar 0.0 -r $OUT_DIR/ref.rttm -s $OUT_DIR/sys.rttm >$OUT_DIR/result_full
fi
