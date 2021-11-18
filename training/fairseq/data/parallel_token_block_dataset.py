# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from fairseq.data import FairseqDataset, plasma_utils, TokenBlockDataset


class ParallelTokenBlockDataset(TokenBlockDataset):
    """
    Wrapper around the TokenBlockDataset class for parallel denoising
    """

    def __init__(
        self,
        src_dataset,
        tgt_dataset,
        src_sizes,
        tgt_sizes,
        block_size,
        pad,
        eos,
        break_mode=None,
        document_sep_len=1,
        src_eos=None,
        tgt_eos=None,
        src_bos=None,
        tgt_bos=None,
        shuffle=False,
    ):
        try:
            from fairseq.data.token_block_utils_fast import (
                _get_slice_indices_fast,
                _get_block_to_dataset_index_fast,
            )
        except ImportError:
            raise ImportError(
                "Please build Cython components with: `pip install --editable .` "
                "or `python setup.py build_ext --inplace`"
            )

        self.src_dataset = src_dataset
        self.tgt_dataset = tgt_dataset
        self.pad = pad
        self.eos = eos

        self.src_eos = src_eos
        self.tgt_eos = tgt_eos
        self.src_bos = src_bos
        self.tgt_bos = tgt_bos

        assert len(src_dataset) == len(src_sizes)
        assert len(tgt_dataset) == len(tgt_sizes)
        assert len(src_dataset) == len(tgt_dataset)
        assert len(src_dataset) > 0

        if isinstance(tgt_sizes, list):
            tgt_sizes = np.array(tgt_sizes, dtype=np.int64)
        else:
            if torch.is_tensor(tgt_sizes):
                tgt_sizes = tgt_sizes.numpy()
            tgt_sizes = tgt_sizes.astype(np.int64)

        if isinstance(src_sizes, list):
            src_sizes = np.array(src_sizes, dtype=np.int64)
        else:
            if torch.is_tensor(src_sizes):
                src_sizes = src_sizes.numpy()
            src_sizes = src_sizes.astype(np.int64)

        break_mode = break_mode if break_mode is not None else "none"

        # For "eos" break-mode, block_size is not required parameters.
        if break_mode == "eos" and block_size is None:
            block_size = 0

        max_sizes = np.maximum(src_sizes, tgt_sizes)
        self.shuffle = shuffle
        if self.shuffle:
            orig_indices = np.arange(len(src_dataset))
            shuffled_indices = orig_indices.copy()
            np.random.shuffle(shuffled_indices)
            self.mapping = {
                "fwd": {
                    orig_indices[i]: shuffled_indices[i]
                    for i in range(len(src_dataset))
                },
                "rev": {
                    shuffled_indices[i]: orig_indices[i]
                    for i in range(len(src_dataset))
                },
            }
            new_sizes = np.arange(len(src_dataset))
            for key in self.mapping["rev"]:
                new_sizes[key] = max_sizes[self.mapping["rev"][key]]
            max_sizes = new_sizes

        slice_indices = _get_slice_indices_fast(
            max_sizes,
            str(break_mode),
            block_size,
            document_sep_len,
        )
        self._sizes = slice_indices[:, 1] - slice_indices[:, 0]

        # build index mapping block indices to the underlying dataset indices
        if break_mode == "eos":
            # much faster version for eos break mode
            block_to_dataset_index = np.stack(
                [
                    np.arange(len(src_sizes)),  # starting index in dataset
                    np.zeros(
                        len(src_sizes), dtype=np.long
                    ),  # starting offset within starting index
                    np.arange(len(src_sizes)),  # ending index in dataset
                ],
                1,
            )
        else:
            block_to_dataset_index = _get_block_to_dataset_index_fast(
                max_sizes,
                slice_indices,
            )
        self._slice_indices = plasma_utils.PlasmaArray(slice_indices)
        self._block_to_dataset_index = plasma_utils.PlasmaArray(block_to_dataset_index)
        additional_src = 0
        additional_tgt = 0
        if self.src_eos is not None:
            additional_src += 1
        if self.src_bos is not None:
            additional_src += 1

        if self.tgt_eos is not None:
            additional_tgt += 1
        if self.tgt_bos is not None:
            additional_tgt += 1
        additional_tokens = max(additional_src, additional_tgt)
        self._sizes += additional_tokens
        self._sizes = plasma_utils.PlasmaArray(self._sizes)

    def __getitem__(self, index):
        start_ds_idx, start_offset, end_ds_idx = self.block_to_dataset_index[index]

        slice_s, slice_e = self.slice_indices[index]
        length = slice_e - slice_s
        s, e = start_offset, start_offset + length

        if self.shuffle:
            src_buffer = torch.cat(
                [
                    self.src_dataset[self.mapping["rev"][idx]]
                    for idx in range(start_ds_idx, end_ds_idx + 1)
                ]
            )
            tgt_buffer = torch.cat(
                [
                    self.tgt_dataset[self.mapping["rev"][idx]]
                    for idx in range(start_ds_idx, end_ds_idx + 1)
                ]
            )

        else:
            src_buffer = torch.cat(
                [self.src_dataset[idx] for idx in range(start_ds_idx, end_ds_idx + 1)]
            )
            tgt_buffer = torch.cat(
                [self.tgt_dataset[idx] for idx in range(start_ds_idx, end_ds_idx + 1)]
            )

        src_item = src_buffer[s:e]
        tgt_item = tgt_buffer[s:e]

        if self.src_bos is not None:
            src_item = torch.cat([src_item.new([self.src_bos]), src_item])
        if self.src_eos is not None:
            src_item = torch.cat([src_item, src_item.new([self.src_eos])])
        if self.tgt_bos is not None:
            tgt_item = torch.cat([tgt_item.new([self.tgt_bos]), tgt_item])
        if self.tgt_eos is not None:
            tgt_item = torch.cat([tgt_item, tgt_item.new([self.tgt_eos])])
        return src_item, tgt_item
