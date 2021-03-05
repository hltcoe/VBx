#!/usr/bin/env python

# @Authors: Lukas Burget, Mireia Diez, Federico Landini, Jan Profant
# @Emails: burget@fit.vutbr.cz, mireia@fit.vutbr.cz, landini@fit.vutbr.cz, jan.profant@phonexia.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# The recipe consists in doing Agglomerative Hierachical Clustering on
# x-vectors in a first step. Then, Variational Bayes HMM over x-vectors
# is applied using the AHC output as args.initialization.
#
# A detailed analysis of this approach is presented in
# M. Diez, L. Burget, F. Landini, S. Wang, J. \v{C}ernock\'{y}
# Optimizing Bayesian HMM based x-vector clustering for the second DIHARD speech
# diarization challenge, ICASSP 2020
# A more thorough description and study of the VB-HMM with eigen-voice priors
# approach for diarization is presented in
# M. Diez, L. Burget, F. Landini, J. \v{C}ernock\'{y}
# Analysis of Speaker Diarization based on Bayesian HMM with Eigenvoice Priors,
# IEEE Transactions on Audio, Speech and Language Processing, 2019
# 
# TODO: Add new paper

import argparse
import os
import itertools
import collections

import h5py
import kaldi_io
import numpy as np
from scipy.special import softmax
from scipy.linalg import eigh

from diarization_lib import read_xvector_timing_dict, l2_norm, cos_similarity, twoGMMcalib_lin, AHC, \
    merge_adjacent_labels, mkdir_p
from kaldi_utils import read_plda
from VB_diarization import VB_diarization

import xvectors.gen_embed as coe_xvec_gen_embed
import em_gmm_clean

import logging
logger = logging.getLogger(__name__)


def write_output(fp, out_labels, starts, ends):
    for label, seg_start, seg_end in zip(out_labels, starts, ends):
        fp.write(f'SPEAKER {file_name} 1 {seg_start:03f} {seg_end - seg_start:03f} '
                 f'<NA> <NA> {label + 1} <NA> <NA>{os.linesep}')


def read_plda(plda_file, plda_format):
    if plda_format == 'kaldi':
        kaldi_plda = read_plda(plda_file)
        plda_mu, plda_tr, plda_psi = kaldi_plda
        W = np.linalg.inv(plda_tr.T.dot(plda_tr))
        B = np.linalg.inv((plda_tr.T / plda_psi).dot(plda_tr))
        acvar, wccn = eigh(B, W)
        plda_psi = acvar[::-1]
        plda_tr = wccn.T[::-1]
        # plda_mu.shape = (128,)
        # plda_psi.shape = (128,)
        # plda_tr.shape = (128, 128)
    elif plda_format == 'pytorch':
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        Ulda, d_wc, d_ac = coe_xvec_gen_embed.get_plda(plda_file)
        # Ulda.shape = (128, 128)
        # d_wc.shape = (128,)
        # d_ac.shape = (128,)
        # rename these variables to what is expected in the code
        # TODO: verify this!!!
        #  1 - if I define plda_tr = Ulda, in the example wav file, I get a DER of 8.12
        #  2 - if I define plda_tr = Ulda.T, in the example wav file, I get a DER of 2.14 !!
        plda_tr = Ulda.T
        plda_psi = d_ac
        plda_mu = np.zeros_like(d_ac)

    return plda_tr, plda_psi, plda_mu


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--init', required=True, type=str, choices=['AHC', 'AHC+VB', 'GMM', 'AHC+GMM'],
                        help='AHC for using only AHC or AHC+VB for VB-HMM after AHC initilization', )
    parser.add_argument('--out-rttm-dir', required=True, type=str, help='Directory to store output rttm files')
    parser.add_argument('--xvec-ark-file', required=True, type=str, nargs='+',
                        help='Kaldi ark file with x-vectors from one or more input recordings. '
                             'Specify multiple files to indicate multi-pass diarization.'
                             'Attention: all x-vectors from one recording must be in one ark file')
    parser.add_argument('--segments-file', required=True, type=str, nargs='+',
                        help='File with x-vector timing info (see diarization_lib.read_xvector_timing_dict)'
                             'Specify multiple files to indicate multi-pass diarization')
    parser.add_argument('--xvec-transform', required=False, type=str,
                        help='path to x-vector transformation h5 file')
    parser.add_argument('--plda-file', required=True, type=str, nargs='+',
                        help='File with PLDA model in Kaldi format used for AHC and VB-HMM x-vector clustering.'
                             'Specify multiple files to indicate multi-pass diarization.'
                             'WARNING!! All PLDA models provided are assumed to be of the same format!')
    parser.add_argument('--plda-format', required=False, type=str, default='kaldi', choices=['kaldi', 'pytorch'],
                        help='Format of stored PLDA, must be either kaldi or pytorch')
    parser.add_argument('--threshold', required=True, type=float, help='args.threshold (bias) used for AHC')
    parser.add_argument('--lda-dim', required=False, type=int,
                        help='For VB-HMM, x-vectors are reduced to this dimensionality using LDA')
    parser.add_argument('--Fa', required=True, type=float,
                        help='Parameter of VB-HMM (see VB_diarization.VB_diarization)')
    parser.add_argument('--Fb', required=True, type=float,
                        help='Parameter of VB-HMM (see VB_diarization.VB_diarization)')
    parser.add_argument('--loopP', required=True, type=float,
                        help='Parameter of VB-HMM (see VB_diarization.VB_diarization)')
    parser.add_argument('--target-energy', required=False, type=float, default=1.0,
                        help='Parameter affecting AHC if the similarity matrix is obtained with PLDA. '
                             '(see diarization_lib.kaldi_ivector_plda_scoring_dense)')
    parser.add_argument('--init-smoothing', required=False, type=float, default=5.0,
                        help='AHC produces hard assignments of x-vetors to speakers. These are "smoothed" to soft '
                             'assignments as the args.initialization for VB-HMM. This parameter controls the amount of'
                             ' smoothing. Not so important, high value (e.g. 10) is OK  => keeping hard assigment')
    parser.add_argument('--output-2nd', required=False, type=bool, default=False,
                        help='Output also second most likely speaker of VB-HMM')

    args = parser.parse_args()
    assert 0 <= args.loopP <= 1, f'Expecting loopP between 0 and 1, got {args.loopP} instead.'

    ###########
    #### Some input validation
    # require the LDA dimension only if the xvec-transform is defined
    if args.xvec_transform is not None:
        if args.lda_dim is None:
            raise ValueError("lda-dim must be defined if xvec-transform is defined!")

    if args.xvec_transform is None and args.plda_format == 'kaldi':
        logger.warning('xvec_transform is None but plda-format is set to `kaldi`! '
                       'Proceeding, but did you forget to set plda-format to `pytorch`?')

    # check if multi-pass is defined
    num_xvecs_defined = len(args.xvec_ark_file)
    num_segs_defined = len(args.segments_file)
    num_pldas_defined = len(args.plda_file)
    if not (num_xvecs_defined == num_segs_defined == num_pldas_defined):
        raise ValueError("Number of xvector files, segments, and pldas must be equal!")
    num_diarization_passes = num_xvecs_defined

    recoid2labels_nthpass_1stmostlikely = {}
    recoid2labels_nthpass_2ndmostlikely = {}
    recoid2numpasses = collections.defaultdict(int)
    for diarization_pass_ii in range(num_diarization_passes):
        # segments file with x-vector timing information
        segs_dict = read_xvector_timing_dict(args.segments_file[diarization_pass_ii])
        # read the plda
        plda_tr, plda_psi, plda_mu = read_plda(args.plda_file[diarization_pass_ii], args.plda_format)
        # Open ark file with x-vectors and in each iteration of the following for-loop
        # read a batch of x-vectors corresponding to one recording
        arkit = kaldi_io.read_vec_flt_ark(args.xvec_ark_file[diarization_pass_ii])
        recit = itertools.groupby(arkit, lambda e: e[0].rsplit('_', 1)[0]) # group xvectors in ark by recording name
        for ii, (file_name, segs) in enumerate(recit):
            #logger.info(ii, file_name)
            print(ii, file_name, diarization_pass_ii)
            seg_names, xvecs = zip(*segs)
            x = np.array(xvecs)

            if args.xvec_transform is not None:
                with h5py.File(args.xvec_transform, 'r') as f:
                    mean1 = np.array(f['mean1'])
                    mean2 = np.array(f['mean2'])
                    lda = np.array(f['lda'])
                    x = l2_norm(lda.T.dot((l2_norm(x - mean1)).transpose()).transpose() - mean2)

            labels1st = recoid2labels_nthpass_1stmostlikely.get(file_name, None)
            if args.init == 'AHC' or args.init.endswith('VB') or args.init.endswith('GMM'):
                if args.init.startswith('AHC'):
                    # Kaldi-like AHC of x-vectors (scr_mx is matrix of pairwise
                    # similarities between all x-vectors)
                    scr_mx = cos_similarity(x)
                    # Figure out utterance specific args.threshold for AHC.
                    thr, junk = twoGMMcalib_lin(scr_mx.ravel())
                    # output "labels" is an integer vector of speaker (cluster) ids
                    labels1st = AHC(scr_mx, thr + args.threshold)

                if args.init.endswith('VB'):
                    # Smooth the hard labels obtained from AHC to soft assignments
                    # of x-vectors to speakers
                    qinit = np.zeros((len(labels1st), np.max(labels1st) + 1))
                    qinit[range(len(labels1st)), labels1st] = 1.0
                    qinit = softmax(qinit * args.init_smoothing, axis=1)
                    fea = (x - plda_mu).dot(plda_tr.T)
                    # Use VB-HMM for x-vector clustering. Instead of i-vector extractor model, we use PLDA
                    # => GMM with only 1 component, V derived accross-class covariance,
                    # and iE is inverse within-class covariance (i.e. identity)
                    if args.lda_dim is not None:
                        fea = fea[:,:args.lda_dim]
                        # Default - BUT algorithm
                        sm = np.zeros(args.lda_dim)
                        siE = np.ones(args.lda_dim)
                        sV = np.sqrt(plda_psi[:args.lda_dim])
                    else:
                        sm = plda_mu
                        siE = np.ones_like(sm)
                        sV = np.sqrt(plda_psi)

                    q, sp, L = VB_diarization(
                        fea, sm, np.diag(siE), np.diag(sV),
                        pi=None, gamma=qinit, maxSpeakers=qinit.shape[1],
                        maxIters=40, epsilon=1e-6,
                        loopProb=args.loopP, Fa=args.Fa, Fb=args.Fb)

                    labels1st = np.argsort(-q, axis=1)[:, 0]
                    if q.shape[1] > 1:
                        labels2nd = np.argsort(-q, axis=1)[:, 1]
                        recoid2labels_nthpass_2ndmostlikely[file_name] = labels2nd
                elif args.init.endswith('GMM'):
                    if labels1st is None:
                        M = 7
                    else:
                        M = len(np.unique(labels1st))
                    fea = (x - plda_mu).dot(plda_tr.T)[:, :args.lda_dim]
                    #labels1st = em_gmm_clean(x.T, W, B, M=M, r=0.9, num_iter=30, init_labels=labels1st)
                    labels1st = em_gmm_clean.em_gmm_clean(fea.T, np.ones(args.lda_dim), plda_psi[:args.lda_dim],
                                                          M=M, r=0.9, num_iter=30, init_labels=labels1st)
            else:
                raise ValueError('Wrong option for args.initialization.')

            recoid2numpasses[file_name] += 1
            recoid2labels_nthpass_1stmostlikely[file_name] = labels1st

            if recoid2numpasses[file_name] == num_diarization_passes:
                assert(np.all(segs_dict[file_name][0] == np.array(seg_names)))
                start, end = segs_dict[file_name][1].T

                labels_1stmostlikely_nthpass = recoid2labels_nthpass_1stmostlikely[file_name]

                starts, ends, out_labels = merge_adjacent_labels(start, end, labels_1stmostlikely_nthpass)
                mkdir_p(args.out_rttm_dir)
                with open(os.path.join(args.out_rttm_dir, f'{file_name}.rttm'), 'w') as fp:
                    write_output(fp, out_labels, starts, ends)

                if args.output_2nd and args.init.endswith('VB') and q.shape[1] > 1:
                    labels_2ndmostlikely_nthpass = recoid2labels_nthpass_2ndmostlikely[filename]
                    starts, ends, out_labels2 = merge_adjacent_labels(start, end, labels_2ndmostlikely_nthpass)
                    output_rttm_dir = f'{args.out_rttm_dir}2nd'
                    mkdir_p(output_rttm_dir)
                    with open(os.path.join(output_rttm_dir, f'{file_name}.rttm'), 'w') as fp:
                        write_output(fp, out_labels2, starts, ends)

                # TODO: delete the keys from recoid2* dicts, b/c we no longer need to keep them
