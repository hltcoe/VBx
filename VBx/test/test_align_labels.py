#!/usr/bin/env python

import unittest
import tempfile
import os
from VBx.vbhmm import align_labels

import numpy as np

class TestAlignLabels(unittest.TestCase):
    def write_segments_to_segfile(self, segments, fout):
        with open(fout, 'w') as f:
            for segment in segments:
                f.write('%s %s %0.02f %0.02f\n' %
                        (segment['id'], segment['spkr'], segment['t_start'], segment['t_end']))

    def test_nonovlp(self):
        # note that id and spkr are no-ops
        cfg1_segments = [
            {'id': 'a', 'spkr': 'a', 't_start': 0.0, 't_end': 1.0},
            {'id': 'a', 'spkr': 'a', 't_start': 1.0, 't_end': 2.0},
            {'id': 'a', 'spkr': 'a', 't_start': 2.0, 't_end': 3.0}
        ]
        cfg2_segments = [
            {'id': 'a', 'spkr': 'a', 't_start': 0.0, 't_end': 0.25},
            {'id': 'a', 'spkr': 'a', 't_start': 0.25, 't_end': 0.50},
            {'id': 'a', 'spkr': 'a', 't_start': 0.5, 't_end': 0.75},
            {'id': 'a', 'spkr': 'a', 't_start': 0.75, 't_end': 1.0},
            {'id': 'a', 'spkr': 'a', 't_start': 1.0, 't_end': 1.25},
            {'id': 'a', 'spkr': 'a', 't_start': 1.25, 't_end': 1.50},
            {'id': 'a', 'spkr': 'a', 't_start': 1.5, 't_end': 1.75},
            {'id': 'a', 'spkr': 'a', 't_start': 1.75, 't_end': 2.0},
            {'id': 'a', 'spkr': 'a', 't_start': 2.0, 't_end': 2.25},
            {'id': 'a', 'spkr': 'a', 't_start': 2.25, 't_end': 2.50},
            {'id': 'a', 'spkr': 'a', 't_start': 2.5, 't_end': 2.75},
            {'id': 'a', 'spkr': 'a', 't_start': 2.75, 't_end': 3.0},
        ]

        labels_cfg1 = np.zeros(len(cfg1_segments))
        for ii in range(len(labels_cfg1)):
            labels_cfg1[ii] = ii

        _, f1 = tempfile.mkstemp()
        _, f2 = tempfile.mkstemp()
        self.write_segments_to_segfile(cfg1_segments, f1)
        self.write_segments_to_segfile(cfg2_segments, f2)

        labels_cfg2 = align_labels(labels_cfg1, f1, f2)
        expected_labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        self.assertTrue(np.allclose(labels_cfg2, expected_labels))

        os.remove(f1)
        os.remove(f2)

    def test_ovlp(self):
        # note that id and spkr are no-ops
        cfg1_segments = [
            {'id': 'a', 'spkr': 'a', 't_start': 0.0, 't_end': 1.0},  # id=0
            {'id': 'a', 'spkr': 'a', 't_start': 1.0, 't_end': 2.0},  # id=1
            {'id': 'a', 'spkr': 'a', 't_start': 2.0, 't_end': 3.0}   # id=2
        ]
        cfg2_segments = [
            {'id': 'a', 'spkr': 'a', 't_start': 0.1, 't_end': 1.1},   # 0
            {'id': 'a', 'spkr': 'a', 't_start': 0.6, 't_end': 1.2},   # 0
            {'id': 'a', 'spkr': 'a', 't_start': 0.6, 't_end': 1.5},   # 1
        ]

        labels_cfg1 = np.zeros(len(cfg1_segments))
        for ii in range(len(labels_cfg1)):
            labels_cfg1[ii] = ii

        _, f1 = tempfile.mkstemp()
        _, f2 = tempfile.mkstemp()
        self.write_segments_to_segfile(cfg1_segments, f1)
        self.write_segments_to_segfile(cfg2_segments, f2)

        labels_cfg2 = align_labels(labels_cfg1, f1, f2)
        expected_labels = np.asarray([0, 0, 1])
        self.assertTrue(np.allclose(labels_cfg2, expected_labels))

        os.remove(f1)
        os.remove(f2)

if __name__ == '__main__':
    unittest.main()