#!/bin/bash

#num_iter_vec=(30 100 300 1000)
#M_vec=(4 5 6 7)
#r_vec=(0.85 0.9 0.92 0.94 0.96 0.98 0.999 1.0)
num_iter_vec=(30)
M_vec=(7)
r_vec=(0.9)
N0_vec=(5 10 15 20 25 30 35 40 45 50 75 100 200)

num_passes=2
model_type="wb"
if ((num_passes==1)); then
    echo "Running 1-pass!"
    ## CAREFUL TO POINT TO THE DIRECTORY that corresponds to the model you used for extracting xvecs!!
    if [[ $model_type == "nb" ]]; then
        sweep_results="/exp/kkarra/diarization/vbx/dihard2019_sweep_results/1pass_dev_nb_new"
        exp_root="/exp/kkarra/diarization/vbx/dihard_2019/exp_COE_2pass_dev_nb_nb_new"  # use the 2-pass dir, b/c this has the NB extractor @ 2 sec for first pass
    elif [[ $model_type == "wb" ]]; then
        # use the 1-pass dir for the WB extractor!
        sweep_results="/exp/kkarra/diarization/vbx/dihard2019_sweep_results/1pass_dev_wb"
        exp_root="/exp/kkarra/diarization/vbx/dihard_2019/exp_COE_1pass_dev_wb"
    fi
else
    echo "Running 2-pass!"
    sweep_results="/exp/kkarra/diarization/vbx/dihard2019_sweep_results/2pass_dev_wb_wb_N02ndpass"
    #exp_root="/exp/kkarra/diarization/vbx/dihard_2019/exp_COE_2pass_dev_nb_nb_new"
    exp_root="/exp/kkarra/diarization/vbx/dihard_2019/exp_COE_2pass_dev_wb_wb"
fi
mkdir -p $sweep_results

N0_firstpass=30  # chosen greedily by first sweeping N0 when fixing N0 for first pass and second pass

for num_iter in "${num_iter_vec[@]}"; do
    for M in "${M_vec[@]}"; do
        for r in "${r_vec[@]}"; do
            for N0 in "${N0_vec[@]}"; do
                #output_f=$sweep_results/result_fair_"$num_iter"_"$M"_"$r"_"$N0"
                output_f=$sweep_results/result_fair_"$num_iter"_"$M"_"$r"_"$N0_firstpass"_"$N0"
                if [ ! -f $output_f ]; then
                    echo "Running num_iter=$num_iter M=$M r=$r N0=$N0"
                    # remove the old diarization run
                    rm -rf $exp_root/out_dir_GMM/

                    # sweeping over N0 in first pass only
                    #./DIHARD2_run_coexvecplda.sh diarization GMM $exp_root $exp_root/xvectors $exp_root/../audio_wav/dev $exp_root/../flist_dev.txt $exp_root/../sad_dev $exp_root/../dihard2_dev.rttm $num_passes all.q $num_iter $M $r $N0 $N0

                    # sweep over N0 in second pass only
                    ./DIHARD2_run_coexvecplda.sh diarization GMM $exp_root $exp_root/xvectors $exp_root/../audio_wav/dev $exp_root/../flist_dev.txt $exp_root/../sad_dev $exp_root/../dihard2_dev.rttm $num_passes all.q $num_iter $M $r $N0_firstpass $N0

                    # copy result into sweep_results
                    cp $exp_root/out_dir_GMM/result_fair $output_f
                fi
            done
        done
    done
done
