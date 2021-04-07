#!/bin/bash

num_pass=2
num_pass_xvec_extract=2  # leave this at 2, it will extract xvectors for 2 passes, but you can still run
                         # diarization at just 1 pass

niter=30
M=7
r=0.9
N0_firstpass=250000
N0_secondpass=250000
kmeans_only=0

reextract_xvectors=0


# CALLHOME Split2
#exp_root=/exp/kkarra/diarization/vbx/callhome/exp_COE_2pass_kaldi_coe_nb_nb_new; 
#if [[ $reextract_xvectors -ge 1 || -f $exp_root/xvectors_1 || -f $exp_root/xvectors_2  ]]; then
#    rm -rf $exp_root
#    ./CALLHOME_run_coexvecplda.sh xvectors GMM $exp_root $exp_root/xvectors $exp_root/../audio_wav $exp_root/../flist.txt $exp_root/../sad_labels  $exp_root/../fullref.rttm $num_pass_xvec_extract all.q $niter $M $r $N0_firstpass $N0_secondpass
#fi
#rm -rf $exp_root/out_dir_GMM
#./CALLHOME_run_coexvecplda.sh diarization GMM $exp_root $exp_root/xvectors $exp_root/../audio_wav $exp_root/../sre2000-key/clsp/callhome2.list $exp_root/../sad_labels  $exp_root/../sre2000-key/clsp/callhome2.ref.rttm $num_pass all.q $niter $M $r $N0_firstpass $N0_secondpass $kmeans_only

# CALLHOME ALL
exp_root=/exp/kkarra/diarization/vbx/callhome/exp_COE_2pass_kaldi_coe_nb_nb; 
if [[ $reextract_xvectors -ge 1 || -f $exp_root/xvectors_1 || -f $exp_root/xvectors_2  ]]; then
    rm -rf $exp_root
    ./CALLHOME_run_coexvecplda.sh xvectors GMM $exp_root $exp_root/xvectors $exp_root/../audio_wav $exp_root/../flist.txt $exp_root/../sad_labels  $exp_root/../fullref.rttm $num_pass_xvec_extract all.q $niter $M $r $N0_firstpass $N0_secondpass
fi
rm -rf $exp_root/out_dir_GMM
./CALLHOME_run_coexvecplda.sh diarization GMM $exp_root $exp_root/xvectors $exp_root/../audio_wav $exp_root/../flist.txt $exp_root/../sad_labels  $exp_root/../fullref.rttm $num_pass all.q $niter $M $r $N0_firstpass $N0_secondpass $kmeans_only


: '
# Dihard2 - Dev
exp_root=/exp/kkarra/diarization/vbx/dihard_2019/exp_COE_2pass_dev_nb_nb_new; 
if [[ $reextract_xvectors -ge 1 || -f $exp_root/xvectors_1 || -f $exp_root/xvectors_2  ]]; then
    rm -rf $exp_root
    ./DIHARD2_run_coexvecplda.sh xvectors GMM $exp_root $exp_root/xvectors $exp_root/../audio_wav/dev $exp_root/../flist_dev.txt $exp_root/../sad_dev $exp_root/../dihard2_dev.rttm $num_pass_xvec_extract all.q $niter $M $r $N0_firstpass $N0_secondpass
fi
rm -rf $exp_root/out_dir_GMM
./DIHARD2_run_coexvecplda.sh diarization GMM $exp_root $exp_root/xvectors $exp_root/../audio_wav/dev $exp_root/../flist_dev.txt $exp_root/../sad_dev $exp_root/../dihard2_dev.rttm $num_pass all.q $niter $M $r $N0_firstpass $N0_secondpass $kmeans_only

# Dihard2 - Eval
exp_root=/exp/kkarra/diarization/vbx/dihard_2019/exp_COE_2pass_eval_nb_nb_new; 
if [[ $reextract_xvectors -ge 1 || -f $exp_root/xvectors_1 || -f $exp_root/xvectors_2  ]]; then
    rm -rf $exp_root
    ./DIHARD2_run_coexvecplda.sh xvectors GMM $exp_root $exp_root/xvectors $exp_root/../audio_wav/eval $exp_root/../flist_eval.txt $exp_root/../sad_eval $exp_root/../dihard2_eval.rttm $num_pass_xvec_extract all.q $niter $M $r $N0_firstpass $N0_secondpass
fi
rm -rf $exp_root/out_dir_GMM
./DIHARD2_run_coexvecplda.sh diarization GMM $exp_root $exp_root/xvectors $exp_root/../audio_wav/eval $exp_root/../flist_eval.txt $exp_root/../sad_eval $exp_root/../dihard2_eval.rttm $num_pass all.q $niter $M $r $N0_firstpass $N0_secondpass $kmeans_only

# AMI - Dev
exp_root=/exp/kkarra/diarization/vbx/ami/exp_COE_2pass_DEV_onlywords_nb_nb_new;
if [[ $reextract_xvectors -ge 1 || -f $exp_root/xvectors_1 || -f $exp_root/xvectors_2  ]]; then
    rm -rf $exp_root
    ./AMI_run_coexvecplda.sh xvectors GMM $exp_root $exp_root/xvectors $exp_root/../audio_wav $exp_root/../AMI-diarization-setup/lists/dev.meetings.txt $exp_root/../AMI-diarization-setup/only_words/labs/dev/ $exp_root/../AMI-diarization-setup/only_words/rttms/dev/ $num_pass_xvec_extract all.q $niter $M $r $N0_firstpass $N0_secondpass
fi
rm -rf $exp_root/out_dir_GMM
./AMI_run_coexvecplda.sh diarization GMM $exp_root $exp_root/xvectors $exp_root/../audio_wav $exp_root/../AMI-diarization-setup/lists/dev.meetings.txt $exp_root/../AMI-diarization-setup/only_words/labs/dev/ $exp_root/../AMI-diarization-setup/only_words/rttms/dev/ $num_pass all.q $niter $M $r $N0_firstpass $N0_secondpass $kmeans_only

# AMI - Test
exp_root=/exp/kkarra/diarization/vbx/ami/exp_COE_2pass_TEST_onlywords_nb_nb_new
if [[ $reextract_xvectors -ge 1 || -f $exp_root/xvectors_1 || -f $exp_root/xvectors_2  ]]; then
    rm -rf $exp_root
    ./AMI_run_coexvecplda.sh xvectors GMM $exp_root $exp_root/xvectors $exp_root/../audio_wav $exp_root/../AMI-diarization-setup/lists/test.meetings.txt $exp_root/../AMI-diarization-setup/only_words/labs/test/ $exp_root/../AMI-diarization-setup/only_words/rttms/test/ $num_pass_xvec_extract all.q $niter $M $r $N0_firstpass $N0_secondpass
fi
rm -rf $exp_root/out_dir_GMM
./AMI_run_coexvecplda.sh diarization GMM $exp_root $exp_root/xvectors $exp_root/../audio_wav $exp_root/../AMI-diarization-setup/lists/test.meetings.txt $exp_root/../AMI-diarization-setup/only_words/labs/test/ $exp_root/../AMI-diarization-setup/only_words/rttms/test/ $num_pass all.q $niter $M $r $N0_firstpass $N0_secondpass $kmeans_only


# AMI-Headset - Dev
exp_root=/exp/kkarra/diarization/vbx/ami-headset/exp_COE_2pass_nb_nb_DEV_onlywords
if [[ $reextract_xvectors -ge 1 || -f $exp_root/xvectors_1 || -f $exp_root/xvectors_2  ]]; then
    rm -rf $exp_root
    ./AMI_run_coexvecplda.sh xvectors GMM $exp_root $exp_root/xvectors $exp_root/../audio_wav $exp_root/../AMI-diarization-setup/lists/dev.meetings.txt $exp_root/../AMI-diarization-setup/only_words/labs/dev/ $exp_root/../AMI-diarization-setup/only_words/rttms/dev/ $num_pass_xvec_extract all.q $niter $M $r $N0_firstpass $N0_secondpass
fi
rm -rf $exp_root/out_dir_GMM
./AMI_run_coexvecplda.sh diarization GMM $exp_root $exp_root/xvectors $exp_root/../audio_wav $exp_root/../AMI-diarization-setup/lists/dev.meetings.txt $exp_root/../AMI-diarization-setup/only_words/labs/dev/ $exp_root/../AMI-diarization-setup/only_words/rttms/dev/ $num_pass all.q $niter $M $r $N0_firstpass $N0_secondpass $kmeans_only

# AMI-Headset - Test
exp_root=/exp/kkarra/diarization/vbx/ami-headset/exp_COE_2pass_nb_nb_TEST_onlywords
if [[ $reextract_xvectors -ge 1 || -f $exp_root/xvectors_1 || -f $exp_root/xvectors_2  ]]; then
    rm -rf $exp_root
    ./AMI_run_coexvecplda.sh xvectors GMM $exp_root $exp_root/xvectors $exp_root/../audio_wav $exp_root/../AMI-diarization-setup/lists/test.meetings.txt $exp_root/../AMI-diarization-setup/only_words/labs/test/ $exp_root/../AMI-diarization-setup/only_words/rttms/test/ $num_pass_xvec_extract all.q $niter $M $r $N0_firstpass $N0_secondpass
fi
rm -rf $exp_root/out_dir_GMM
./AMI_run_coexvecplda.sh diarization GMM $exp_root $exp_root/xvectors $exp_root/../audio_wav $exp_root/../AMI-diarization-setup/lists/test.meetings.txt $exp_root/../AMI-diarization-setup/only_words/labs/test/ $exp_root/../AMI-diarization-setup/only_words/rttms/test/ $num_pass all.q $niter $M $r $N0_firstpass $N0_secondpass $kmeans_only
'

