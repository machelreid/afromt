# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import numpy as np
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    DenoisingDataset,
    ParallelDenoisingDataset,
    ParallelTokenBlockDataset,
    MultilingualDenoisingDataset,
    Dictionary,
    PrependTokenDataset,
    ResamplingDataset,
    SortDataset,
    TruncateDataset,
    StripTokenDataset,
    TokenBlockDataset,
    LanguagePairDataset,
    encoders,
    indexed_dataset,
    data_utils,
)
from fairseq.data.encoders.utils import get_whole_word_mask
from fairseq.tasks import register_task

import itertools

from .denoising import DenoisingTask
from .translation import TranslationTask


logger = logging.getLogger(__name__)


def get_all_pairs(pairs):
    out_list = []
    for p in pairs:
        src, tgt = p.split("-")
        out_list.append(src + "-" + tgt)
        out_list.append(tgt + "-" + src)
    return out_list


def load_langpair_dataset_legacy(
    data_path,
    split,
    src,
    src_dict,
    tgt,
    tgt_dict,
    combine,
    dataset_impl,
    upsample_primary,
    left_pad_source,
    left_pad_target,
    max_source_positions,
    max_target_positions,
    prepend_bos=False,
    load_alignments=False,
    truncate_source=False,
    append_source_id=False,
    num_buckets=0,
    shuffle=True,
    pad_to_multiple=1,
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, data_path)
                )

        src_dataset = data_utils.load_indexed_dataset(
            prefix + src, src_dict, dataset_impl
        )
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(
            prefix + tgt, tgt_dict, dataset_impl
        )
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        logger.info(
            "{} {} {}-{} {} examples".format(
                data_path, split_k, src, tgt, len(src_datasets[-1])
            )
        )

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())

    eos = None
    if append_source_id:
        # not for us!!!
        # src_dataset = AppendTokenDataset(
        #     src_dataset, src_dict.index("[{}]".format(src))
        # )
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(
                tgt_dataset, tgt_dict.index("[{}]".format(tgt))
            )
        eos = tgt_dict.index("[{}]".format(tgt))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, "{}.align.{}-{}".format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(
                align_path, None, dataset_impl
            )

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    return LanguagePairDataset(
        src_dataset,
        src_dataset.sizes,
        src_dict,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset,
        eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
    )


def load_langpair_dataset(
    args,
    data_path,
    split,
    src,
    src_dict,
    tgt,
    tgt_dict,
    combine,
    dataset_impl,
    upsample_primary,
    left_pad_source,
    left_pad_target,
    max_source_positions,
    max_target_positions,
    seed,
    mask_idx,
    mask_whole_words,
    prepend_bos=False,
    load_alignments=False,
    truncate_source=False,
    append_source_id=False,
    num_buckets=0,
    shuffle=True,
    pad_to_multiple=1,
    return_early=False,
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, data_path)
                )

        src_dataset = data_utils.load_indexed_dataset(
            prefix + src, src_dict, dataset_impl
        )
        # src_dataset = StripTokenDataset(src_dataset, src_dict.eos())
        if truncate_source:
            src_dataset = TruncateDataset(
                src_dataset,
                max_source_positions - 1,
            )
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(
            prefix + tgt, tgt_dict, dataset_impl
        )
        if tgt_dataset is not None:
            # tgt_dataset = StripTokenDataset(tgt_dataset, tgt_dict.eos())
            tgt_datasets.append(tgt_dataset)

        logger.info(
            "{} {} {}-{} {} examples".format(
                data_path, split_k, src, tgt, len(src_datasets[-1])
            )
        )

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None

    if return_early:
        tgt_dataset = PrependTokenDataset(
            tgt_dataset, tgt_dict.index("[{}]".format(tgt))
        )
        return src_dataset, tgt_dataset
    parallel_dataset = ParallelTokenBlockDataset(
        src_dataset,
        tgt_dataset,
        src_dataset.sizes,
        tgt_dataset_sizes,
        args.tokens_per_sample - 2,
        pad=src_dict.pad(),
        eos=src_dict.eos(),
        break_mode="complete",
        document_sep_len=0,
        src_eos=src_dict.eos(),
        tgt_eos=tgt_dict.index("[{}]".format(tgt)),
        src_bos=src_dict.bos(),
        tgt_bos=tgt_dict.bos(),
    )
    logger.info(
        "loaded {} blocks from: {}".format(
            len(parallel_dataset), split + "." + src + "-" + tgt
        )
    )
    return ParallelDenoisingDataset(
        parallel_dataset,
        parallel_dataset.sizes,
        src_dict,  # under the !!!very presumptuous!!! assumption that src_dict == tgt_dict
        mask_idx,
        mask_whole_words,
        shuffle=args.shuffle_instance,
        seed=seed,
        args=args,
    )


@register_task("denoising_translation")
class MultilingualDenoisingTranslationTask(DenoisingTask):
    @staticmethod
    def add_args(parser):
        DenoisingTask.add_args(parser)
        parser.add_argument(
            "--load-alignments",
            action="store_true",
            help="load the binarized alignments",
        )
        parser.add_argument(
            "--left-pad-source",
            default="True",
            type=str,
            metavar="BOOL",
            help="pad the source on the left",
        )
        parser.add_argument(
            "--left-pad-target",
            default="False",
            type=str,
            metavar="BOOL",
            help="pad the target on the left",
        )
        parser.add_argument(
            "--upsample-primary",
            default=1,
            type=int,
            help="amount to upsample primary dataset",
        )
        parser.add_argument(
            "--denoising-sampling-alpha",
            type=float,
            default=1.0,
            help="smoothing alpha for sample ratios across multiple denoising datasets",
        )
        parser.add_argument(
            "--translation-sampling-alpha",
            type=float,
            default=1.0,
            help="smoothing alpha for sample ratios across multiple translation datasets",
        )
        parser.add_argument(
            "--denoising-translation-sampling-alpha",
            type=float,
            default=1.0,
            help="smoothing alpha for ratios across denoising and translating",
        )
        parser.add_argument("--add-lang-token", default=False, action="store_true")
        parser.add_argument(
            "--langs", type=str, help="language ids we are considering", default=None
        )
        parser.add_argument(
            "--translation-pairs",
            type=str,
            help="comma-seperated list of language pairs we are considering for translation",
            default=None,
        )
        parser.add_argument(
            "--no-whole-word-mask-langs",
            type=str,
            default="",
            metavar="N",
            help="languages without spacing between words dont support whole word masking",
        )
        parser.add_argument(
            "--parallel-language-mixing",
            action="store_true",
            help="[SRC -> TGT] <ja> japanese <ar> arabic <zh> chinese -> English",
        )

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task."""
        paths = args.data.split(":")
        assert len(paths) > 0
        dictionary = Dictionary.load(
            os.path.join(paths[0], "denoising", args.langs.split(",")[0], "dict.txt")
        )

        data_path = paths[0]
        if args.langs is None:
            languages = sorted(
                [
                    name
                    for name in os.listdir(os.path.join(data_path, "denoising"))
                    if os.path.isdir(os.path.join(data_path, "denoising", name))
                ]
            )
        else:
            languages = args.langs.split(",")

        if args.add_lang_token:
            dictionary.add_symbol("[NEU]")
            for lang in languages:
                dictionary.add_symbol("[{}]".format(lang))

        logger.info("dictionary: {} types".format(len(dictionary)))
        if not hasattr(args, "shuffle_instance"):
            args.shuffle_instance = False
        return cls(args, dictionary)

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)
        self.dictionary = dictionary
        self.seed = args.seed

        # add mask token
        self.mask_idx = self.dictionary.add_symbol("<mask>")
        self.langs = args.langs
        self.translation_pairs = args.translation_pairs
        self.args = args

    def _get_sample_prob(self, dataset_lens, task="denoising", translation_pairs=None):
        """
        Get smoothed sampling probability by languages. This helps low resource
        languages by upsampling them.
        """
        if translation_pairs is not None:
            new_dataset_lens = []
            count_dict = {}
            tgt_langs = [i.split("-")[1] for i in translation_pairs]
            set_tgts = set(tgt_langs)
            for t in set_tgts:
                count_dict[t] = min(tgt_langs.count(t), 2)
            for i, pair in enumerate(translation_pairs):
                tgt = pair.split("-")[1]
                new_dataset_lens.append(dataset_lens[i] // count_dict[tgt])
            dataset_lens = np.array(new_dataset_lens)

        prob = dataset_lens / dataset_lens.sum()
        smoothed_prob = prob ** getattr(self.args, task + "_sampling_alpha", 0.3)
        smoothed_prob = smoothed_prob / smoothed_prob.sum()
        return smoothed_prob

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.args.data.split(":")
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]
        split_path = os.path.join(data_path, split)

        if self.langs is None:
            languages = sorted(
                [
                    name
                    for name in os.listdir(os.path.join(data_path, "denoising"))
                    if os.path.isdir(os.path.join(data_path, "denoising", name))
                ]
            )
        else:
            languages = self.langs.split(",")
            for name in languages:
                p = os.path.join(data_path, "denoising", name)
                assert os.path.exists(p), "data not found: {}".format(p)

        if self.translation_pairs is None:
            translation_pairs = sorted(
                [
                    name
                    for name in os.listdir(os.path.join(data_path, "translation"))
                    if os.path.isdir(os.path.join(data_path, "translation", name))
                ]
            )
        else:
            translation_pairs = self.translation_pairs.split(",")
            for name in translation_pairs:
                p = os.path.join(data_path, "translation", name)
                assert os.path.exists(p), "data not found: {}".format(p)

        logger.info("Denosing on {0} languages: {1}".format(len(languages), languages))
        logger.info(
            "Translating on {0} language pairs: {1}".format(
                len(translation_pairs) * 2, get_all_pairs(translation_pairs)
            )
        )
        logger.info(
            "Language to id mapping: ", {lang: id for id, lang in enumerate(languages)}
        )

        mask_whole_words = get_whole_word_mask(self.args, self.dictionary)
        language_without_segmentations = self.args.no_whole_word_mask_langs.split(",")
        lang_datasets = []
        for language in languages:
            split_path = os.path.join(data_path, "denoising", language, split)

            dataset = data_utils.load_indexed_dataset(
                split_path,
                self.source_dictionary,
                self.args.dataset_impl,
                combine=combine,
            )
            if dataset is None:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, split_path)
                )

            end_token = self.source_dictionary.index("[NEU]")

            # create continuous blocks of tokens
            dataset = TokenBlockDataset(
                dataset,
                dataset.sizes,
                self.args.tokens_per_sample - 2,  # one less for <s>
                pad=self.source_dictionary.pad(),
                eos=end_token,
                break_mode=self.args.sample_break_mode,
            )
            logger.info("loaded {} blocks from: {}".format(len(dataset), split_path))

            # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
            dataset = PrependTokenDataset(dataset, self.source_dictionary.bos())
            dataset = AppendTokenDataset(dataset, end_token)

            lang_mask_whole_words = (
                mask_whole_words
                if language not in language_without_segmentations
                else None
            )
            lang_dataset = MultilingualDenoisingDataset(
                dataset,
                dataset.sizes,
                self.dictionary,
                self.mask_idx,
                lang_mask_whole_words,
                shuffle=self.args.shuffle_instance,
                seed=self.seed,
                args=self.args,
                eos=end_token,
            )
            lang_datasets.append(lang_dataset)

        translation_datasets = []
        for pair in translation_pairs:

            split_path = os.path.join(data_path, "translation", pair)

            for i in range(2):
                if i == 0:
                    src, tgt = pair.split("-")
                elif i == 1:
                    tgt, src = pair.split("-")

                lang_mask_whole_words = (
                    mask_whole_words
                    if src not in language_without_segmentations
                    else None
                )
                translation_dataset = load_langpair_dataset(
                    self.args,
                    split_path,
                    split,
                    src,
                    self.dictionary,
                    tgt,
                    self.dictionary,
                    combine=combine,
                    dataset_impl=self.args.dataset_impl,
                    upsample_primary=1.0,  # self.args.upsample_primary,
                    left_pad_source=self.args.left_pad_source,
                    left_pad_target=self.args.left_pad_target,
                    max_source_positions=getattr(
                        self.args, "max_source_positions", 1024
                    ),
                    max_target_positions=getattr(
                        self.args, "max_target_positions", 1024
                    ),
                    seed=self.seed,
                    mask_idx=self.mask_idx,
                    mask_whole_words=lang_mask_whole_words,
                    load_alignments=self.args.load_alignments,
                    prepend_bos=True,  # Keeping in fashion with the mBART training
                    append_source_id=True,
                    return_early=True if self.args.parallel_language_mixing else False,
                )

                if dataset is None:
                    raise FileNotFoundError(
                        "Dataset not found: {} ({})".format(split, split_path)
                    )
                translation_datasets.append(translation_dataset)

        lang_dataset_lengths = np.array(
            [len(d) for d in lang_datasets],
            dtype=float,
        )
        if self.args.parallel_language_mixing:
            translation_dataset_lengths = np.array(
                [len(d) for d in translation_datasets],
                dtype=float,
            )
        else:
            translation_dataset_lengths = np.array(
                [len(d[0]) for d in translation_datasets],
                dtype=float,
            )
        logger.info(
            "loaded total {} blocks for all languages".format(
                int(lang_dataset_lengths.sum()),
            )
        )
        if split == self.args.train_subset:
            # For train subset, additionally up or down sample languages.
            sample_probs = self._get_sample_prob(lang_dataset_lengths, task="denoising")
            logger.info(
                "Sample probability by language: {}".format(
                    {
                        lang: "{0:.4f}".format(sample_probs[id])
                        for id, lang in enumerate(languages)
                    }
                )
            )
            size_ratio = (
                sample_probs * lang_dataset_lengths.sum()
            ) / lang_dataset_lengths
            logger.info(
                "Up/Down Sampling ratio by language: {}".format(
                    {
                        lang: "{0:.2f}".format(size_ratio[id])
                        for id, lang in enumerate(languages)
                    }
                )
            )

            resampled_lang_datasets = [
                ResamplingDataset(
                    lang_datasets[i],
                    size_ratio=size_ratio[i],
                    seed=self.args.seed,
                    epoch=epoch,
                    replace=size_ratio[i] >= 1.0,
                )
                for i, d in enumerate(lang_datasets)
            ]
            denoising_dataset = ConcatDataset(resampled_lang_datasets)

            sample_probs = self._get_sample_prob(
                translation_dataset_lengths,
                task="translation",
                translation_pairs=get_all_pairs(translation_pairs),
            )
            logger.info(
                "Sample probability by translation pair: {}".format(
                    {
                        pair: "{0:.4f}".format(sample_probs[id])
                        for id, pair in enumerate(get_all_pairs(translation_pairs))
                    }
                )
            )
            size_ratio = (
                sample_probs * translation_dataset_lengths.sum()
            ) / translation_dataset_lengths
            logger.info(
                "Up/Down Sampling ratio by translation pair: {}".format(
                    {
                        pair: "{0:.4f}".format(size_ratio[id])
                        for id, pair in enumerate(get_all_pairs(translation_pairs))
                    }
                )
            )
            if self.args.parallel_language_mixing:
                resampled_translation_datasets = [
                    [
                        ResamplingDataset(
                            translation_datasets[i][idx],
                            size_ratio=size_ratio[i],
                            seed=self.args.seed,
                            epoch=epoch,
                            replace=size_ratio[i] >= 1.0,
                        )
                        for idx in range(2)
                    ]
                    for i, d in enumerate(translation_datasets)
                ]
                translation_dataset_src = ConcatDataset(
                    [i[0] for i in resampled_translation_datasets]
                )
                translation_dataset_tgt = ConcatDataset(
                    [i[1] for i in resampled_translation_datasets]
                )
                translation_dataset = ParallelTokenBlockDataset(
                    translation_dataset_src,
                    translation_dataset_tgt,
                    translation_dataset_src.sizes,
                    translation_dataset_tgt.sizes,
                    self.args.tokens_per_sample - 2,
                    pad=self.dictionary.pad(),
                    eos=self.dictionary.eos(),
                    break_mode="complete",
                    document_sep_len=0,
                    src_eos=self.dictionary.eos(),
                    tgt_eos=self.dictionary.eos(),
                    src_bos=self.dictionary.bos(),
                    tgt_bos=self.dictionary.bos(),
                )
                logger.info(
                    "loaded {} blocks from: {}".format(
                        len(translation_dataset), split + "." + src + "-" + tgt
                    )
                )

                translation_dataset = ParallelDenoisingDataset(
                    translation_dataset,
                    translation_dataset.sizes,
                    self.dictionary,  # under the !!!very presumptuous!!! assumption that src_dict == tgt_dict
                    self.mask_idx,
                    get_whole_word_mask(self.args, self.dictionary),
                    shuffle=self.args.shuffle_instance,
                    seed=self.args.seed,
                    args=self.args,
                )
            else:
                resampled_translation_datasets = [
                    ResamplingDataset(
                        translation_datasets[i],
                        size_ratio=size_ratio[i],
                        seed=self.args.seed,
                        epoch=epoch,
                        replace=size_ratio[i] >= 1.0,
                    )
                    for i, d in enumerate(translation_datasets)
                ]

                translation_dataset = ConcatDataset(resampled_translation_datasets)

            datasets = [denoising_dataset, translation_dataset]
            dataset_lengths = np.array(
                [len(denoising_dataset), len(translation_dataset)], dtype=float
            )
            sample_probs = self._get_sample_prob(
                dataset_lengths, task="denoising_translation"
            )
            logger.info(
                "Sample probability by task: {}".format(
                    {
                        pair: "{0:.4f}".format(sample_probs[id])
                        for id, pair in enumerate(["denoising", "translation"])
                    }
                )
            )
            size_ratio = (sample_probs * dataset_lengths.sum()) / dataset_lengths
            logger.info(
                "Up/Down Sampling ratio by task: {}".format(
                    {
                        pair: "{0:.4f}".format(size_ratio[id])
                        for id, pair in enumerate(["denoising", "translation"])
                    }
                )
            )
            resampled_datasets = [
                ResamplingDataset(
                    datasets[i],
                    size_ratio=size_ratio[i],
                    seed=self.args.seed,
                    epoch=epoch,
                    replace=size_ratio[i] >= 1.0,
                )
                for i, d in enumerate(datasets)
            ]
            dataset = ConcatDataset(resampled_datasets)

        else:
            if self.args.parallel_language_mixing:
                translation_dataset_src = ConcatDataset(
                    [i[0] for i in translation_datasets]
                )
                translation_dataset_tgt = ConcatDataset(
                    [i[1] for i in translation_datasets]
                )
                translation_dataset = ParallelTokenBlockDataset(
                    translation_dataset_src,
                    translation_dataset_tgt,
                    translation_dataset_src.sizes,
                    translation_dataset_tgt.sizes,
                    self.args.tokens_per_sample - 2,
                    pad=self.dictionary.pad(),
                    eos=self.dictionary.eos(),
                    break_mode="complete",
                    document_sep_len=0,
                    src_eos=self.dictionary.eos(),
                    tgt_eos=self.dictionary.eos(),
                    src_bos=self.dictionary.bos(),
                    tgt_bos=self.dictionary.bos(),
                    shuffle=True,
                )
                logger.info(
                    "loaded {} blocks from: {}".format(
                        len(translation_dataset), split + "." + src + "-" + tgt
                    )
                )
                translation_dataset = ParallelDenoisingDataset(
                    translation_dataset,
                    translation_dataset.sizes,
                    self.dictionary,  # under the !!!very presumptuous!!! assumption that src_dict == tgt_dict
                    self.mask_idx,
                    get_whole_word_mask(self.args, self.dictionary),
                    shuffle=self.args.shuffle_instance,
                    seed=self.args.seed,
                    args=self.args,
                )

            else:
                translation_dataset = ConcatDataset(translation_datasets)
            dataset = ConcatDataset([ConcatDataset(lang_datasets), translation_dataset])
            lang_splits = [split]
            for lang_id, lang_dataset in enumerate(lang_datasets):
                split_name = split + "_" + languages[lang_id]
                lang_splits.append(split_name)
                self.datasets[split_name] = lang_dataset
            for pair_id, translation_dataset in enumerate(translation_datasets):
                split_name = split + "_" + get_all_pairs(translation_pairs)[pair_id]
                lang_splits.append(split_name)
                self.datasets[split_name] = translation_dataset

            if split in self.args.valid_subset:
                self.args.valid_subset = self.args.valid_subset.replace(
                    split, ",".join(lang_splits)
                )

        with data_utils.numpy_seed(self.args.seed + epoch):
            shuffle = np.random.permutation(len(dataset))

        self.datasets[split] = SortDataset(
            dataset,
            sort_order=[
                shuffle,
                dataset.sizes,
            ],
        )
