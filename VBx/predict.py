#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Authors: Lukas Burget, Federico Landini, Jan Profant
# @Emails: burget@fit.vutbr.cz, landini@fit.vutbr.cz, jan.profant@phonexia.com

import argparse
import logging
import os
import time

import kaldi_io
import numpy as np
import onnxruntime
import soundfile as sf
import torch.backends

import features
from models.resnet import *

# comes from pykaldi - which is only installable on linux
from kaldi.util.table import SequentialMatrixReader
import tempfile
import sys
import uuid

torch.backends.cudnn.enabled = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()
        if self.name:
            logger.info(f'Start: {self.name}: ')

    def __exit__(self, type, value, traceback):
        if self.name:
            logger.info(f'End:   {self.name}: Elapsed: {time.time() - self.tstart} seconds')
        else:
            logger.info(f'End:   {self.name}: ')


def initialize_gpus(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus


def load_utt(ark, utt, position):
    with open(ark, 'rb') as f:
        f.seek(position - len(utt) - 1)
        ark_key = kaldi_io.read_key(f)
        assert ark_key == utt, f'Keys does not match: `{ark_key}` and `{utt}`.'
        mat = kaldi_io.read_mat(f)
        return mat


def write_txt_vectors(path, data_dict):
    """ Write vectors file in text format.

    Args:
        path (str): path to txt file
        data_dict: (Dict[np.array]): name to array mapping
    """
    with open(path, 'w') as f:
        for name in sorted(data_dict):
            f.write(f'{name}  [ {" ".join(str(x) for x in data_dict[name])} ]{os.linesep}')


def get_embedding(fea, model, label_name=None, input_name=None, backend='pytorch'):
    if backend == 'pytorch':
        data = torch.from_numpy(fea).to(device)
        data = data[None, :, :]
        data = torch.transpose(data, 1, 2)
        spk_embeds = model(data)
        return spk_embeds.data.cpu().numpy()[0]
    elif backend == 'onnx':
        return model.run([label_name],
                         {input_name: fea.astype(np.float32).transpose()
                         [np.newaxis, :, :]})[0].squeeze()


def parse_kaldi_cfg(cfg_in):
    with open(cfg_in, 'r') as f:
        all_lines = f.readlines()
    # remove trailing new-lines and remove comments
    all_lines_cleaned = [x.rstrip().split('#')[0] for x in all_lines]
    # convert to dictionary
    cfg_dict = {}
    for l in all_lines_cleaned:
        l_split = l.split('=')
        k = l_split[0].split('--')[1]
        v = l_split[1].rstrip()
        if v == 'true':
            cfg_dict[k] = True
        elif v == 'false':
            cfg_dict[k] = False
        else:
            cfg_dict[k] = int(v)
    return cfg_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default='', help='use gpus (passed to CUDA_VISIBLE_DEVICES)')
    parser.add_argument('--model', required=False, type=str, default=None, help='name of the model')
    parser.add_argument('--weights', required=True, type=str, default=None, help='path to pretrained model weights')
    parser.add_argument('--model-file', required=False, type=str, default=None, help='path to model file')
    parser.add_argument('--ndim', required=False, type=int, default=64, help='dimensionality of features')
    parser.add_argument('--embed-dim', required=False, type=int, default=256, help='dimensionality of the emb')
    parser.add_argument('--seg-len', required=False, type=int, default=144, help='segment length')
    parser.add_argument('--seg-jump', required=False, type=int, default=24, help='segment jump')
    parser.add_argument('--in-file-list', required=True, type=str, help='input list of files')
    parser.add_argument('--in-lab-dir', required=True, type=str, help='input directory with VAD labels')
    parser.add_argument('--in-wav-dir', required=True, type=str, help='input directory with wavs')
    parser.add_argument('--out-ark-fn', required=True, type=str, help='output embedding file')
    parser.add_argument('--out-seg-fn', required=True, type=str, help='output segments file')
    parser.add_argument('--backend', required=False, default='pytorch', choices=['pytorch', 'onnx'],
                        help='backend that is used for x-vector extraction')

    parser.add_argument('--feat-extract-engine', required=False, default='but', choices=['but', 'kaldi'],
                        help='Which engine to use for feature extraction')
    parser.add_argument('--kaldi-feat-extract-conf', required=False, default=None,
                        help='.conf file to configure Kaldi feature extraction')
    parser.add_argument('--kaldi-root-dir', required=False, type=str, default=None,
                        help='Kaldi installation root directory')
    parser.add_argument('--kaldi-fbank-conf', required=False, type=str, default=None,
                        help='Configuration to extract filterbank features')

    args = parser.parse_args()

    seg_len = args.seg_len
    seg_jump = args.seg_jump

    device = ''
    if args.gpus != '':
        logger.info(f'Using GPU: {args.gpus}')

        # gpu configuration
        initialize_gpus(args)
        device = torch.device(device='cuda')
    else:
        device = torch.device(device='cpu')

    model, label_name, input_name = '', None, None

    if args.backend == 'pytorch':
        if args.model_file is not None:
            model = torch.load(args.model_file)
            model = model.to(device)
        elif args.model is not None and args.weights is not None:
            model = eval(args.model)(feat_dim=args.ndim, embed_dim=args.embed_dim)
            model = model.to(device)
            checkpoint = torch.load(args.weights, map_location=device)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            model.eval()
    elif args.backend == 'onnx':
        model = onnxruntime.InferenceSession(args.weights)
        input_name = model.get_inputs()[0].name
        label_name = model.get_outputs()[0].name

    else:
        raise ValueError('Wrong combination of --model/--weights/--model_file '
                         'parameters provided (or not provided at all)')

    file_names = np.atleast_1d(np.loadtxt(args.in_file_list, dtype=object))

    with torch.no_grad():
        with open(args.out_seg_fn, 'w') as seg_file:
            with open(args.out_ark_fn, 'wb') as ark_file:
                for fn in file_names:
                    with Timer(f'Processing file {fn}'):
                        signal, samplerate = sf.read(f'{os.path.join(args.in_wav_dir, fn)}.wav')
                        labs = np.atleast_2d((np.loadtxt(f'{os.path.join(args.in_lab_dir, fn)}.lab',
                                                         usecols=(0, 1)) * samplerate).astype(int))
                        if samplerate == 8000:
                            noverlap = 120
                            winlen = 200
                            window = features.povey_window(winlen)
                            fbank_mx = features.mel_fbank_mx(
                                winlen, samplerate, NUMCHANS=64, LOFREQ=20.0, HIFREQ=3700, htk_bug=False)
                        elif samplerate == 16000:
                            noverlap = 240
                            winlen = 400
                            window = features.povey_window(winlen)
                            fbank_mx = features.mel_fbank_mx(
                                winlen, samplerate, NUMCHANS=64, LOFREQ=20.0, HIFREQ=7600, htk_bug=False)
                        else:
                            raise ValueError(f'Only 8kHz and 16kHz are supported. Got {samplerate} instead.')

                        LC = 150
                        RC = 149

                        np.random.seed(3)  # for reproducibility
                        signal = features.add_dither((signal * 2 ** 15).astype(int))

                        if args.feat_extract_engine.lower() == 'kaldi':
                            # add Kaldi to the sys path
                            if args.kaldi_root_dir is None:
                                raise ValueError("kaldi-root-dir must be specified if using "
                                                 "Kaldi feature extraction engine")
                            if args.kaldi_fbank_conf is None:
                                raise ValueError("kaldi-fbank-conf must be specified if using "
                                                 "Kaldi feature extraction engine")
                            sys.path.append(os.path.join(args.kaldi_root_dir, 'src', 'featbin'))

                            # create a temporary wav.scp for Kaldi to process the feats file
                            f_obj, wav_scp_fname = tempfile.mkstemp(suffix='.scp', prefix='wav_', text=True)
                            tmp_sid = str(uuid.uuid4())
                            with open(wav_scp_fname, 'w') as ff:
                                ff.write('%s %s' % (tmp_sid, f'{os.path.abspath(os.path.join(args.in_wav_dir, fn))}.wav'))
                            os.close(f_obj)

                            # specify the Kaldi command to extract the features
                            kaldi_cmd = "ark:compute-fbank-feats --config=%s scp:%s ark:- | " \
                                        "apply-cmvn-sliding --norm-vars=false --cmn-window=300 --center=true ark:- ark:- |" % \
                                        (args.kaldi_fbank_conf, wav_scp_fname)

                            # extract the feature, store in memory, and verify length of features matches the
                            # required configuration
                            with SequentialMatrixReader(kaldi_cmd) as f:
                                # NOTE: a for-loop is required for using pykaldi (as far as i can tell), but
                                #  it only runs one iteration
                                for (feats_key, feat) in f:
                                    assert feats_key == tmp_sid
                                    feats_kaldi_all = np.asarray(feat)
                                    break
                                # ensure size of feature file is what we expect
                                n_feats, feat_dim = feats_kaldi_all.shape
                                fbank_config_dict = parse_kaldi_cfg(args.kaldi_fbank_conf)
                                # 10ms kaldi default, units in ms implied by Kaldi
                                frameshift_ms = fbank_config_dict.get('frame-shift', 10)

                                frameshift_samps = int(frameshift_ms/1000. * samplerate)
                                expected_nfeats = len(signal)/frameshift_samps
                                nfeats_tol = 2
                                assert abs(n_feats-expected_nfeats) < nfeats_tol, \
                                    "Number of generated features is not within expected range!"

                            # delete the temporary wav.scp file
                            os.remove(wav_scp_fname)

                        for segnum in range(len(labs)):
                            seg = signal[labs[segnum, 0]:labs[segnum, 1]]
                            if seg.shape[0] > 0.01 * samplerate:  # process segment only if longer than 0.01s
                                if args.feat_extract_engine.lower() == 'but':
                                    # Mirror noverlap//2 initial and final samples
                                    seg = np.r_[seg[noverlap // 2 - 1::-1],
                                                seg, seg[-1:-winlen // 2 - 1:-1]]
                                    fea = features.fbank_htk(seg, window, noverlap, fbank_mx, USEPOWER=True,
                                                             ZMEANSOURCE=True)
                                    fea = features.cmvn_floating_kaldi(fea, LC, RC, norm_vars=False).astype(np.float32)
                                elif args.feat_extract_engine.lower() == 'kaldi':
                                    t_start = labs[segnum, 0]   # in units of seconds
                                    t_stop = labs[segnum, 1]    # in units of seconds
                                    start_ii = int(np.floor(t_start*samplerate))
                                    frameshift_s = frameshift_ms/1000.
                                    n_slices = int(np.ceil((t_stop-t_start)/frameshift_s))
                                    fea = feats_kaldi_all[:, start_ii:start_ii+n_slices]

                                slen = len(fea)
                                start = -seg_jump

                                for start in range(0, slen - seg_len, seg_jump):
                                    data = fea[start:start + seg_len]
                                    xvector = get_embedding(
                                        data, model, label_name=label_name, input_name=input_name, backend=args.backend)

                                    key = f'{fn}_{segnum:04}-{start:08}-{(start + seg_len):08}'
                                    if np.isnan(xvector).any():
                                        logger.warning(f'NaN found, not processing: {key}{os.linesep}')
                                    else:
                                        seg_file.write(f'{key} {fn} '
                                                       f'{round(labs[segnum, 0] / float(samplerate) + start / 100.0, 3)} '
                                                       f'{round(labs[segnum, 0] / float(samplerate) + start / 100.0 + seg_len / 100.0, 3)}'
                                                       f'{os.linesep}')
                                        kaldi_io.write_vec_flt(ark_file, xvector, key=key)

                                if slen - start - seg_jump >= 10:
                                    data = fea[start + seg_jump:slen]
                                    xvector = get_embedding(
                                        data, model, label_name=label_name, input_name=input_name, backend=args.backend)

                                    key = f'{fn}_{segnum:04}-{(start + seg_jump):08}-{slen:08}'

                                    if np.isnan(xvector).any():
                                        logger.warning(f'NaN found, not processing: {key}{os.linesep}')
                                    else:
                                        seg_file.write(f'{key} {fn} '
                                                       f'{round(labs[segnum, 0] / float(samplerate) + (start + seg_jump) / 100.0, 3)} '
                                                       f'{round(labs[segnum, 1] / float(samplerate), 3)}'
                                                       f'{os.linesep}')
                                        kaldi_io.write_vec_flt(ark_file, xvector, key=key)
