"""
Metrics that have to do with (1) how many segmentations a tokeniser could generate in theory,
and (2) the amount of tokens it generates in practice.
"""
from typing import Tuple, Iterable, Dict, Set, Union
from dataclasses import dataclass
from math import log2

from ..util.iterables import streamProgress
from ..util.types import NamedIterable
from ..interfaces.tokeniser import TokeniserWithVocabDict, Vocab, Tokeniser, Preprocessor


def countValidSegmentations(pretoken: str, vocab: Union[Vocab, Set[str]]) -> int:
    """
    Computes how many possible segmentations a vocabulary allows for a given string. Note: no preprocessor is applied here!
    Forward Viterbi algorithm, which is O(n^2) instead of O(2^n) even though there are O(2^n) segmentations.
    """
    options_to_get_before_char = [0 for _ in range(len(pretoken)+1)]
    options_to_get_before_char[0] = 1
    for i in range(len(pretoken)):
        for j in range(i+1, len(pretoken)+1):
            if pretoken[i:j] in vocab:  # not j+1 because we step to BEFORE character j, so it is an exclusive bound
                options_to_get_before_char[j] += options_to_get_before_char[i]
    return options_to_get_before_char[-1]


def prepareAndCountValidSegmentations(word: str, preprocessor: Preprocessor, vocab: Union[Vocab, Set[str]]) -> Tuple[int, int, int]:
    """
    Note that the vocabulary exists in the output space of a preprocessor. This has two implications for counting segmentations:
        1. You should make sure that the given preprocessor is the full, effective preprocessor used before segmenting
           into tokens of the vocabulary. This is NOT always `tokeniser.preprocessor`, because some tokenisers (esp. those
           from other packages) add characters to the lowest-level pretokens. You need to model that too.
        2. The valid segmentations have to respect boundaries placed by the pretokeniser.
    """
    n_chars = 0
    n_segs  = 1
    pretokens = preprocessor.do(word)
    for pretoken in pretokens:
        n_chars += len(pretoken)
        n_segs  *= countValidSegmentations(pretoken, vocab)
    return n_segs, n_chars, len(pretokens)


@dataclass
class VocabularyFertility:
    """
    How many segmentations a vocabulary supports, normalised by a variety of metrics, averaged across a word corpus,
    weighed (or not) by word frequency.
    The weighted variants are equivalent to computing the unweighted variants but in a corpus where each word type is duplicated by its count.
    """
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


@dataclass
class InferenceFertility:
    """
    Statistics about the segmentations that are actually made in practice by a tokeniser, not how many it hypothetically supports.
    """
    corpus_name: str

    ccc: int  # sum_w f(w)*len(w)
    ctc: int  # sum_w f(w)*len(tk(w))
    cwc: int  # sum_w f(w)

    tokens_per_word_type: float        # [1/sum_w 1] * [sum_w len(tk(w))]
    tokens_per_word_token: float       # [1/sum_w f(w)] * [sum_w f(w)*len(tk(w))]

    chars_per_word_type_token_micro: float   # [sum_w len(w)]/[sum_w len(tk(w))]                or "average token length in a word list"
    chars_per_word_token_token_micro: float  # [sum_w f(w)*len(w)]/[sum_w f(w)*len(tk(w))]      or "average token length in a corpus"
    chars_per_word_type_token_macro: float   # [1/sum_w 1] * [sum_w len(w)/len(tk(w))]          or "average CPT ratio in a word list"
    chars_per_word_token_token_macro: float  # [1/sum_w f(w)] * [sum_w f(w)*len(w)/len(tk(w))]  or "average CPT ratio in a corpus"

    tokens_per_word_type_char_macro: float   # [1/sum_w 1] * [sum_w len(tk(w))/len(w)]          or "average TPC ratio in a word list" (nobody uses this metric since the ratio can never be 0)
    tokens_per_word_token_char_macro: float  # [1/sum_w f(w)] * [sum_w f(w)*len(tk(w))/len(w)]  or "average TPC ratio in a corpus"    (idem)
    segmentality_word_types: float                # [1/sum_w 1] * [sum_w (len(tk(w))-1)/(len(w)-1)]          or "average segmentality in a word list"
    segmentality_word_tokens: float               # [1/sum_w f(w)] * [sum_w f(w)*(len(tk(w))-1)/(len(w)-1)]  or "average segmentality in a corpus"


def getVocabStats(effective_preprocessor: Preprocessor, vocab: Vocab,
                  raw_words: Iterable[str], counts: Dict[str, float]=None,
                  logarithmic_segmentations: bool=False, do_measure_original_word_length: bool=False,
                  exclude_words_over_length: int=100) -> VocabularyFertility:
    """
    Note: if the preprocessor of the tokeniser adds characters that are supposed to be token-building units yet consist
    of multiple characters (like </w>), this function's results are wrong.

    :param logarithmic_segmentations: Because segmentations of a string s scale as O(2^|s|), taking the log means scaling as O(|s|).
    """
    if counts is None:
        counts = dict()

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

    for raw_word in streamProgress(raw_words):
        if len(raw_word) > exclude_words_over_length:
            continue

        segs, chars, n_pretokens = prepareAndCountValidSegmentations(raw_word, effective_preprocessor, vocab)

        # Maximal segmentations are measured in pretoken space.
        max_segs            = 2**(chars-1)  # every position between characters can be split on or not
        max_segs_restricted = 2**(chars-n_pretokens)  # for every extra pretoken, 1 split position is fixed
        if logarithmic_segmentations:
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
        seg_to_maxseg_ratio = segs/max_segs_restricted if chars-n_pretokens != 0 else 1

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

    return VocabularyFertility(
        vocab_size=len(vocab),

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


def getInferenceStats(tokeniser: Tokeniser, examples: NamedIterable[str], example_to_words: Preprocessor=None, word_counts: Dict[str, float]=None,
                      do_measure_original_word_length: bool=False, exclude_words_over_length: int=100) -> InferenceFertility:
    word_iterable = examples if example_to_words is None else examples.flatmap(example_to_words.do)
    if word_counts is None:
        word_counts = dict()

    sum_one          = 0
    sum_one_weighted = 0
    sum_tk          = 0
    sum_tk_weighted = 0  # Equivalent to corpus token count (CTC).

    sum_len          = 0
    sum_len_weighted = 0  # Equivalent to corpus character count (CCC).
    sum_len_on_tk          = 0
    sum_len_on_tk_weighted = 0

    sum_tk_on_len          = 0
    sum_tk_on_len_weighted = 0
    sum_tk1_on_len1          = 0
    sum_tk1_on_len1_weighted = 0

    for word in word_iterable:
        if len(word) == 0 or len(word) > exclude_words_over_length:
            continue

        tokens = tokeniser.prepareAndTokenise(word)
        if len(tokens) == 0:
            continue

        chars = len(word) if do_measure_original_word_length else sum(map(len, tokens))
        tk    = len(tokens)
        char_to_token_ratio = chars/tk
        token_to_char_ratio = tk/chars
        segmentality        = (tk-1)/(chars-1) if chars > 1 else 1.0

        # Accumulate
        f_w = word_counts.get(word, 1)

        sum_one          += 1
        sum_one_weighted += 1*f_w
        sum_tk          += tk
        sum_tk_weighted += tk*f_w
        sum_len          += chars
        sum_len_weighted += chars*f_w
        sum_len_on_tk          += char_to_token_ratio
        sum_len_on_tk_weighted += char_to_token_ratio*f_w
        sum_tk_on_len          += token_to_char_ratio
        sum_tk_on_len_weighted += token_to_char_ratio*f_w
        sum_tk1_on_len1          += segmentality
        sum_tk1_on_len1_weighted += segmentality*f_w

    return InferenceFertility(
        corpus_name=word_iterable.name,

        ccc=sum_len_weighted,
        ctc=sum_tk_weighted,
        cwc=sum_one_weighted,

        tokens_per_word_type=sum_tk/sum_one if word_counts else 0,  # If counts are given, we assume every word appeared exactly once in the iterable. If not, we assume that words can appear multiple times, and hence they are tokens, in which case we know nothing about deduplicated word types.
        tokens_per_word_token=sum_tk_weighted/sum_one_weighted,

        chars_per_word_type_token_micro=sum_len/sum_tk if word_counts else 0,
        chars_per_word_token_token_micro=sum_len_weighted/sum_tk_weighted,

        chars_per_word_type_token_macro=sum_len_on_tk/sum_one if word_counts else 0,
        tokens_per_word_type_char_macro=sum_tk_on_len/sum_one if word_counts else 0,
        chars_per_word_token_token_macro=sum_len_on_tk_weighted/sum_one_weighted,
        tokens_per_word_token_char_macro=sum_tk_on_len_weighted/sum_one_weighted,

        segmentality_word_types=sum_tk1_on_len1/sum_one if word_counts else 0,
        segmentality_word_tokens=sum_tk1_on_len1_weighted/sum_one_weighted
    )
