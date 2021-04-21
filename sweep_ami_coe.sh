#!/bin/bash

#num_iter_vec=(30 100 300 1000)
#M_vec=(4 5 6 7)
#r_vec=(0.85 0.9 0.92 0.94 0.96 0.98 0.999 1.0)
num_iter_vec=(30)
M_vec=(7)
r_vec=(0.9)
N0_vec=(5 10 15 20 25 30 35 40 45 50 75 100 200 500) 

num_passes=2
if ((num_passes==1)); then
    echo "Running 1-pass"
    # CAREFUL TO POINT TO THE DIRECTORY that corresponds to the model you used for extracting xvecs!!
    #sweep_results="/exp/kkarra/diarization/vbx/ami_sweep_results/1pass_dev_nb_new"
    #exp_root="/exp/kkarra/diarization/vbx/ami/exp_COE_2pass_DEV_onlywords_nb_nb_new"  # use the 2-pass dir b/c this has NB extractor @ 2-sec for first pass
    # USE THE 1-pass dir for the WB extractor
    sweep_results="/exp/kkarra/diarization/vbx/ami_sweep_results/1pass_dev_wb"
    exp_root="/exp/kkarra/diarization/vbx/ami/exp_COE_1pass_wb_DEV_onlywords"
else
    echo "Running 2-pass"
    sweep_results="/exp/kkarra/diarization/vbx/ami_sweep_results/2pass_dev_wb_wb_N02ndpass"
    #exp_root="/exp/kkarra/diarization/vbx/ami/exp_COE_2pass_DEV_onlywords_nb_nb_new" # has the NB extractor at 2sec/1sec
    exp_root="/exp/kkarra/diarization/vbx/ami/exp_COE_2pass_DEV_onlywords_wb_wb"
fi
mkdir -p $sweep_results
N0_firstpass=30 # chosen greedily by first sweeping N0

for num_iter in "${num_iter_vec[@]}"; do
    for M in "${M_vec[@]}"; do
        for r in "${r_vec[@]}"; do
            for N0 in "${N0_vec[@]}"; do
                #output_f=$sweep_results/result_forgiving_"$num_iter"_"$M"_"$r"_"$N0"
                output_f=$sweep_results/result_forgiving_"$num_iter"_"$M"_"$r"_"$N0_firstpass"_"$N0"
                if [ ! -f $output_f ]; then
                    echo "Running num_iter=$num_iter M=$M r=$r N0_firstpass=$N0_firstpass N0_secondpass=$N0"
                    # remove the old diarization run
                    rm -rf $exp_root/out_dir_GMM/

                    # sweep N0 w/ same first pass and second pass
                    #./AMI_run_coexvecplda.sh diarization GMM $exp_root $exp_root/xvectors $exp_root/../audio_wav/ $exp_root/../AMI-diarization-setup/lists/dev.meetings.txt $exp_root/../AMI-diarization-setup/only_words/labs/dev/ $exp_root/../AMI-diarization-setup/only_words/rttms/dev/ $num_passes all.q $num_iter $M $r $N0 $N0

                    # sweep N0 on second pass, fix the first pass
                    ./AMI_run_coexvecplda.sh diarization GMM $exp_root $exp_root/xvectors $exp_root/../audio_wav/ $exp_root/../AMI-diarization-setup/lists/dev.meetings.txt $exp_root/../AMI-diarization-setup/only_words/labs/dev/ $exp_root/../AMI-diarization-setup/only_words/rttms/dev/ $num_passes all.q $num_iter $M $r $N0_firstpass $N0

                    # copy result into sweep_results
                    cp $exp_root/out_dir_GMM/result_forgiving $output_f
                fi
            done
        done
    done
done
