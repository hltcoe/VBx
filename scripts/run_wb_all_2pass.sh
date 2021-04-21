#!/bin/bash

dihard_exp_rootdir="/exp/kkarra/diarization/vbx/dihard_2019"
ami_exp_rootdir="/exp/kkarra/diarization/vbx/ami"
ami_headset_exp_rootdir="/exp/kkarra/diarization/vbx/ami-headset"

num_pass=2                    # number of passes to run
num_pass_xvec_extract=2       # number of xvector extracts to perform
                              # recommend leaving this at 2, as it offers flexibility in
                              # understanding the effect of each pass on the diarization error

niter=30                      # num. of update iterations, 2-pass default: 30
M=7                           # num. initial clusters, 2-pass default: 7
r=0.9                         # correlation between samples, 2-pass default: 0.9
N0_firstpass=30               # N0 for the first pass, 2-pass WB default: 30
N0_secondpass=$N0_firstpass   # N0 for the second pass, 2-pass WB default: 30
k_means_only=0                # if set to 1, only runs k-means clustering with no EM updates.
                              # useful for understanding effect of initial clustering on
                              # end-to-end diarization error rate

reextract_xvectors=0          # set to 1 if you want to reextract xvectors
                              # (in case they are stale, model update, etc ...)

# Dihard2 - Dev
exp_root="${dihard_exp_rootdir}/exp_2pass_wb_wb_dev"
if [[ $reextract_xvectors -ge 1 || ! -d $exp_root/xvectors_1 || ! -d $exp_root/xvectors_2  ]]; then
    rm -rf $exp_root
    ../DIHARD2_run_2pass.sh xvectors GMM $exp_root $exp_root/xvectors $exp_root/../audio_wav/dev $exp_root/../flist_dev.txt $exp_root/../sad_dev $exp_root/../dihard2_dev.rttm $num_pass_xvec_extract all.q $niter $M $r $N0_firstpass $N0_secondpass $k_means_only wb
fi
rm -rf $exp_root/out_dir_GMM
../DIHARD2_run_2pass.sh diarization GMM $exp_root $exp_root/xvectors $exp_root/../audio_wav/dev $exp_root/../flist_dev.txt $exp_root/../sad_dev $exp_root/../dihard2_dev.rttm $num_pass all.q $niter $M $r $N0_firstpass $N0_secondpass $k_means_only wb

# Dihard2 - Eval
exp_root="${dihard_exp_rootdir}/exp_2pass_wb_wb_eval"
if [[ $reextract_xvectors -ge 1 || ! -d $exp_root/xvectors_1 || ! -d $exp_root/xvectors_2  ]]; then
    rm -rf $exp_root
    ../DIHARD2_run_2pass.sh xvectors GMM $exp_root $exp_root/xvectors $exp_root/../audio_wav/eval $exp_root/../flist_eval.txt $exp_root/../sad_eval $exp_root/../dihard2_eval.rttm $num_pass_xvec_extract all.q $niter $M $r $N0_firstpass $N0_secondpass $k_means_only wb
fi
rm -rf $exp_root/out_dir_GMM
../DIHARD2_run_2pass.sh diarization GMM $exp_root $exp_root/xvectors $exp_root/../audio_wav/eval $exp_root/../flist_eval.txt $exp_root/../sad_eval $exp_root/../dihard2_eval.rttm $num_pass all.q $niter $M $r $N0_firstpass $N0_secondpass $k_means_only wb

# AMI - Dev
exp_root="${ami_exp_rootdir}/exp_2pass_wb_wb_dev_onlywords"
if [[ $reextract_xvectors -ge 1 || ! -d $exp_root/xvectors_1 || ! -d $exp_root/xvectors_2  ]]; then
    rm -rf $exp_root
    ../AMI_run_2pass.sh xvectors GMM $exp_root $exp_root/xvectors $exp_root/../audio_wav $exp_root/../AMI-diarization-setup/lists/dev.meetings.txt $exp_root/../AMI-diarization-setup/only_words/labs/dev/ $exp_root/../AMI-diarization-setup/only_words/rttms/dev/ $num_pass_xvec_extract all.q $niter $M $r $N0_firstpass $N0_secondpass $k_means_only wb
fi
rm -rf $exp_root/out_dir_GMM
../AMI_run_2pass.sh diarization GMM $exp_root $exp_root/xvectors $exp_root/../audio_wav $exp_root/../AMI-diarization-setup/lists/dev.meetings.txt $exp_root/../AMI-diarization-setup/only_words/labs/dev/ $exp_root/../AMI-diarization-setup/only_words/rttms/dev/ $num_pass all.q $niter $M $r $N0_firstpass $N0_secondpass $k_means_only wb

# AMI - Test
exp_root="${ami_exp_rootdir}/exp_2pass_wb_wb_test_onlywords"
if [[ $reextract_xvectors -ge 1 || ! -d $exp_root/xvectors_1 || ! -d $exp_root/xvectors_2  ]]; then
    rm -rf $exp_root
    ../AMI_run_2pass.sh xvectors GMM $exp_root $exp_root/xvectors $exp_root/../audio_wav $exp_root/../AMI-diarization-setup/lists/test.meetings.txt $exp_root/../AMI-diarization-setup/only_words/labs/test/ $exp_root/../AMI-diarization-setup/only_words/rttms/test/ $num_pass_xvec_extract all.q $niter $M $r $N0_firstpass $N0_secondpass $k_means_only wb
fi
rm -rf $exp_root/out_dir_GMM
../AMI_run_2pass.sh diarization GMM $exp_root $exp_root/xvectors $exp_root/../audio_wav $exp_root/../AMI-diarization-setup/lists/test.meetings.txt $exp_root/../AMI-diarization-setup/only_words/labs/test/ $exp_root/../AMI-diarization-setup/only_words/rttms/test/ $num_pass all.q $niter $M $r $N0_firstpass $N0_secondpass $k_means_only wb

# AMI-Headset - Dev
exp_root="${ami_headset_exp_rootdir}/exp_2pass_wb_wb_dev_onlywords"
if [[ $reextract_xvectors -ge 1 || ! -d $exp_root/xvectors_1 || ! -d $exp_root/xvectors_2  ]]; then
    rm -rf $exp_root
    ../AMI_run_2pass.sh xvectors GMM $exp_root $exp_root/xvectors $exp_root/../audio_wav $exp_root/../AMI-diarization-setup/lists/dev.meetings.txt $exp_root/../AMI-diarization-setup/only_words/labs/dev/ $exp_root/../AMI-diarization-setup/only_words/rttms/dev/ $num_pass_xvec_extract all.q $niter $M $r $N0_firstpass $N0_secondpass $k_means_only wb
fi
rm -rf $exp_root/out_dir_GMM
../AMI_run_2pass.sh diarization GMM $exp_root $exp_root/xvectors $exp_root/../audio_wav $exp_root/../AMI-diarization-setup/lists/dev.meetings.txt $exp_root/../AMI-diarization-setup/only_words/labs/dev/ $exp_root/../AMI-diarization-setup/only_words/rttms/dev/ $num_pass all.q $niter $M $r $N0_firstpass $N0_secondpass $k_means_only wb

# AMI-Headset - Test
exp_root="${ami_headset_exp_rootdir}/exp_2pass_wb_wb_test_onlywords"
if [[ $reextract_xvectors -ge 1 || ! -d $exp_root/xvectors_1 || ! -d $exp_root/xvectors_2  ]]; then
    rm -rf $exp_root
    ../AMI_run_2pass.sh xvectors GMM $exp_root $exp_root/xvectors $exp_root/../audio_wav $exp_root/../AMI-diarization-setup/lists/test.meetings.txt $exp_root/../AMI-diarization-setup/only_words/labs/test/ $exp_root/../AMI-diarization-setup/only_words/rttms/test/ $num_pass_xvec_extract all.q $niter $M $r $N0_firstpass $N0_secondpass $k_means_only wb
fi
rm -rf $exp_root/out_dir_GMM
../AMI_run_2pass.sh diarization GMM $exp_root $exp_root/xvectors $exp_root/../audio_wav $exp_root/../AMI-diarization-setup/lists/test.meetings.txt $exp_root/../AMI-diarization-setup/only_words/labs/test/ $exp_root/../AMI-diarization-setup/only_words/rttms/test/ $num_pass all.q $niter $M $r $N0_firstpass $N0_secondpass $k_means_only wb
