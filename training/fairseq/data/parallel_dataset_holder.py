# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from fairseq.data import FairseqDataset, plasma_utils, TokenBlockDataset


class ParallelDatasetHolder(FairseqDataset):
    """
    Wrapper around the TokenBlockDataset class for parallel denoising
    """

    def __init__(
        self,
        src_dataset,
        tgt_dataset,
    ):
        self.src_dataset = src_dataset
        self.tgt_dataset = tgt_dataset
        self._sizes = np.maximum(src_dataset.sizes, tgt_dataset.sizes)

    def __getitem__(self, index):
        return self.src_dataset[index], self.tgt_dataset[index]

    @property
    def sizes(self):
        return self._sizes

    def __len__(self):
        return len(self.src_dataset)
