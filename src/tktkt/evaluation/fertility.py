"""
Goal: Currently, computes the "segmentations per word" metric.
      In the future, this file can be used for other metrics too, like tokens per word and so on.
"""
from typing import Tuple, Iterable, Dict
from dataclasses import dataclass
from numpy import log2

from ..interfaces.tokeniser import TokeniserWithVocab, Vocab


def possibleSegmentations(vocab: Vocab, pretoken: str) -> int:
    """
    Computes how many possible segmentations a vocabulary allows for a given string.
    Forward Viterbi because ... obviously.
    """
    options_to_get_before_char = [0 for _ in range(len(pretoken)+1)]
    options_to_get_before_char[0] = 1
    for i in range(len(pretoken)):
        for j in range(i+1, len(pretoken)+1):
            if pretoken[i:j] in vocab:  # not j+1 because we step to BEFORE character j, so it is an exclusive bound
                options_to_get_before_char[j] += options_to_get_before_char[i]
    return options_to_get_before_char[-1]


@dataclass
class SegmentationStatistics:
    vocab_size: int

    segmentations_per_word_type: float          # [1/sum_w 1] * [sum_w seg(w)]
    segmentations_per_word_token: float         # [1/sum_w f(w)] * [sum_w f(w)*seg(w)]

    segmentations_per_type_char_micro: float    # [sum_w seg(w)]/[sum_w len(w)]
    segmentations_per_token_char_micro: float   # [sum_w f(w)*seg(w)]/[sum_w f(w)*len(w)]
    segmentations_per_type_char_macro: float    # [1/sum_w 1] * [sum_w seg(w)/len(w)]
    segmentations_per_token_char_macro: float   # [1/sum_w f(w)] * [sum_w f(w)*seg(w)/len(w)]

    segmentations_per_type_max_micro: float    # [sum_w seg(w)]/[sum_w maxseg(w)]
    segmentations_per_token_max_micro: float   # [sum_w f(w)*seg(w)]/[sum_w f(w)*maxseg(w)]
    segmentations_per_type_max_macro: float    # [1/sum_w 1] * [sum_w seg(w)/maxseg(w)]
    segmentations_per_token_max_macro: float   # [1/sum_w f(w)] * [sum_w f(w)*seg(w)/maxseg(w)]


def makeSegmentationStats(prep_and_vocab: TokeniserWithVocab,
                          raw_words: Iterable[str], counts: Dict[str, float]=None,
                          do_measure_original_word_length: bool=False, exclude_words_over_length: int=100, do_log_segmentations: bool=False) -> SegmentationStatistics:
    """
    Note: if the preprocessor of the tokeniser adds characters that are supposed to be token-building units yet consist
    of multiple characters (like </w>), this function's results are wrong.
    """
    if counts is None:
        counts = dict()

    # The "weighted" variants can be computed as unweighted variants in a corpus where each word type is duplicated by its count.
    sum_one          = 0
    sum_one_weighted = 0
    sum_seg          = 0
    sum_seg_weighted = 0

    sum_len          = 0
    sum_len_weighted = 0
    sum_seg_on_len          = 0
    sum_seg_on_len_weighted = 0

    sum_maxseg          = 0
    sum_maxseg_weighted = 0
    sum_seg_on_maxseg          = 0
    sum_seg_on_maxseg_weighted = 0

    for raw_word in raw_words:
        if len(raw_word) > exclude_words_over_length:
            continue

        # Note that the vocabulary exists in the output space of a preprocessor. That means it not only segments
        # strings with characters as in that output space, but it also has to respect boundaries placed by the pretokeniser.
        chars = 0
        segs  = 1
        for pretoken in prep_and_vocab.preprocessor.do(raw_word):
            chars += len(pretoken)
            segs  *= possibleSegmentations(prep_and_vocab.vocab, pretoken)

        # Maximal segmentations are measured in pretoken space.
        max_segs            = 2**(chars-1)  # every position between characters can be split on or not
        max_segs_restricted = 2**(chars-len(prep_and_vocab.preprocessor.do(raw_word)))  # for every extra pretoken, 1 split position is fixed
        if do_log_segmentations:
            if segs == 0:
                print("Word", raw_word, "has no segmentations, so its log is -infinite. Skipping it.")
                continue
            segs                = log2(segs)
            max_segs            = log2(max_segs)
            max_segs_restricted = log2(max_segs_restricted)

        # For metrics that compare against word length, it is possible to work in word space and not pretoken space.
        if do_measure_original_word_length:
            chars = len(raw_word)
        seg_to_char_ratio   = segs/chars
        seg_to_maxseg_ratio = segs/max_segs_restricted

        # Accumulate
        f_w = counts.get(raw_word, 1)

        sum_one          += 1
        sum_one_weighted += 1*f_w
        sum_seg          += segs
        sum_seg_weighted += segs*f_w

        sum_len          += chars
        sum_len_weighted += chars*f_w
        sum_seg_on_len          += seg_to_char_ratio
        sum_seg_on_len_weighted += seg_to_char_ratio*f_w

        sum_maxseg          += max_segs_restricted
        sum_maxseg_weighted += max_segs_restricted*f_w
        sum_seg_on_maxseg          += seg_to_maxseg_ratio
        sum_seg_on_maxseg_weighted += seg_to_maxseg_ratio*f_w
        # print(raw_word, "\t\t", segs, "of", max_segs, "or", max_segs_restricted)

    return SegmentationStatistics(
        vocab_size=len(prep_and_vocab.vocab),

        segmentations_per_word_type =sum_seg/sum_one,
        segmentations_per_word_token=sum_seg_weighted/sum_one_weighted,

        segmentations_per_type_char_micro =sum_seg/sum_len,
        segmentations_per_token_char_micro=sum_seg_weighted/sum_len_weighted,
        segmentations_per_type_char_macro =sum_seg_on_len/sum_one,
        segmentations_per_token_char_macro=sum_seg_on_len_weighted/sum_one_weighted,

        segmentations_per_type_max_micro =sum_seg/sum_maxseg,
        segmentations_per_token_max_micro=sum_seg_weighted/sum_maxseg_weighted,
        segmentations_per_type_max_macro =sum_seg_on_maxseg/sum_one,
        segmentations_per_token_max_macro=sum_seg_on_maxseg_weighted/sum_one_weighted
    )
