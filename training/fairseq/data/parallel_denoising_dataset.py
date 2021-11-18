# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import numpy as np
import torch

from . import FairseqDataset, data_utils, DenoisingDataset


class ParallelDenoisingDataset(DenoisingDataset):
    """
    A wrapper around DenoisingDataset for parallel denoising dataset.
    """

    def __getitem__(self, index):

        with data_utils.numpy_seed(self.seed, self.epoch, index):
            source, target = self.dataset[index]
            assert source[-1] == self.eos
            # assert target[-1] == self.eos
            if self.permute_sentence_ratio > 0.0:
                source = self.permute_sentences(source, self.permute_sentence_ratio)

            if self.mask_ratio > 0:
                source = self.add_whole_word_mask(source, self.mask_ratio)

            if self.insert_ratio > 0:
                source = self.add_insertion_noise(source, self.insert_ratio)

            if self.rotate_ratio > 0.0 and np.random.random() < self.rotate_ratio:
                source = self.add_rolling_noise(source)
        # there can additional changes to make:
        if self.item_transform_func is not None:
            source, target = self.item_transform_func(source, target)

        assert (source >= 0).all()
        assert (source[1:-1] >= 1).all()
        assert (source <= len(self.vocab)).all()
        assert source[0] == self.vocab.bos()
        assert source[-1] == self.eos
        return {
            "id": index,
            "source": source,
            "target": target,
        }
