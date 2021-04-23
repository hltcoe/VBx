#!/bin/bash

#num_iter_vec=(30 100 300 1000)
#M_vec=(4 5 6 7)
#r_vec=(0.85 0.9 0.92 0.94 0.96 0.98 0.999 1.0)
num_iter_vec=(30)
M_vec=(7)
r_vec=(0.9)
N0_vec=(5 10 15 20 25 30 35 40 45 50 75 100 200 500)

sweep_results="/exp/kkarra/diarization/vbx/callhome_sweep_results/2pass_split1_nb_nb_new"
mkdir -p $sweep_results

exp_root="/exp/kkarra/diarization/vbx/callhome/exp_COE_2pass_kaldi_coe_nb_nb_new"

for num_iter in "${num_iter_vec[@]}"; do
    for M in "${M_vec[@]}"; do
        for r in "${r_vec[@]}"; do
            for N0 in "${N0_vec[@]}"; do
                output_f=$sweep_results/result_forgiving_"$num_iter"_"$M"_"$r"_"$N0"
                if [ ! -f $output_f ]; then
                    echo "Running num_iter=$num_iter M=$M r=$r N0=$N0"
                    # remove the old diarization run
                    rm -rf $exp_root/out_dir_GMM/

                    # this is sweeping N0, with fixed N0 for both passes!
                    ./CALLHOME_run_coexvecplda.sh diarization GMM $exp_root $exp_root/xvectors $exp_root/../audio_wav/ $exp_root/../sre2000-key/clsp/callhome1.list $exp_root/../sad_labels $exp_root/../sre2000-key/clsp/callhome1.ref.rttm 2 all.q $num_iter $M $r $N0 $N0

                    # copy result into sweep_results
                    cp $exp_root/out_dir_GMM/result_forgiving $output_f
                fi
            done
        done
    done
done
