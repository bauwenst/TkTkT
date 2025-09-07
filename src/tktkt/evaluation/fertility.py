"""
Metrics that have to do with (1) how many segmentations a tokeniser could generate in theory,
and (2) the amount of tokens it generates in practice.
"""
from pathlib import Path
from typing import Tuple, Set, Union, List
from dataclasses import dataclass

import json
from math import log2
from dacite import from_dict

from ..evaluation.observing import Sent
from ..paths import TkTkTPaths
from ..util.types import Tokens
from ..util.dicts import jsonToDataclass, dataclassToJson
from ..interfaces.tokeniser import Vocab, Preprocessor


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
    ccc: int  # sum_w f(w)*len(w)
    ctc: int  # sum_w f(w)*len(tk(w))
    cwc: int  # sum_w f(w)

    lwc: int  # sum_w 1 or "lexicon word count", i.e. how many unique words occur in the corpus.
    ltc: int  # sum_w len(tk(w))
    lcc: int  # sum_w len(w)

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

    @property
    def mwl(self):
        return self.ccc/self.cwc

    @property
    def mtl(self):
        return self.ccc/self.ctc


from .observing import FinallyObservableObserver, Observer

class PossibleSegmentations(FinallyObservableObserver[Union[str,Tuple[str,int]],VocabularyFertility]):
    """
    Given words (with or without frequency), measures how many possible segmentations are achievable given a vocabulary.
    """

    def __init__(
        self,
        vocab: Vocab, 
        effective_preprocessor: Preprocessor,

        track_unique_words: bool=True,
        do_logarithmic_segmentations: bool=False, 
        do_measure_original_word_length: bool=False,
        exclude_words_over_length: int=100,
        observers: List[Observer[VocabularyFertility]] =None
    ):
        """
        Note: if the preprocessor adds characters that are supposed to be atomic yet consist of multiple characters
        (like </w>), this object's results are wrong.

        :param vocab: set of subword strings used to segment strings concatenatively after ALL preprocessing is applied.
        :param track_unique_words: if True, will output an extra set of metrics wherein every unique word has equal
                                   contribution. This requires storing a set of all seen words, which consumes a lot of memory.
        :param effective_preprocessor: maps raw strings to the space in which the tokeniser splits.
                                       Note that libraries like SentencePiece apply hidden extra preprocessing before splitting!
        :param do_logarithmic_segmentations: Because segmentations of a string s scale as O(2^|s|), taking the log means scaling as O(|s|).
        """
        super().__init__(observers=observers)
        self.vocab                  = vocab
        self.effective_preprocessor = effective_preprocessor
        self._unique_words                    = track_unique_words
        self._do_logarithmic_segmentations    = do_logarithmic_segmentations
        self._do_measure_original_word_length = do_measure_original_word_length
        self._exclude_words_over_length       = exclude_words_over_length

    def _initialiseAsObserver(self, identifier: str):
        self.seen = dict()

        self.sum_one          = 0
        self.sum_one_weighted = 0
        self.sum_seg          = 0
        self.sum_seg_weighted = 0
    
        self.sum_len          = 0
        self.sum_len_weighted = 0
        self.sum_seg_on_len          = 0
        self.sum_seg_on_len_weighted = 0
    
        self.sum_maxseg          = 0
        self.sum_maxseg_weighted = 0
        self.sum_seg_on_maxseg          = 0
        self.sum_seg_on_maxseg_weighted = 0

    def _receive(self, sample: str, weight: float):
        raw_word, f_w = sample, weight
        if len(raw_word) > self._exclude_words_over_length:  # Technically should be done by whichever observable is outputting words to this observer, however, we make it an explicit option because the Viterbi algorithm below can explode otherwise.
            return 

        II = 0
        if self._unique_words and raw_word in self.seen:
            segs, chars, n_pretokens = self.seen[raw_word]
        else:
            segs, chars, n_pretokens = prepareAndCountValidSegmentations(raw_word, self.effective_preprocessor, self.vocab)
            if self._unique_words:
                self.seen[raw_word] = (segs, chars, n_pretokens)
                II = 1

        # Maximal segmentations are measured in pretoken space.
        max_segs            = 2**(chars-1)  # every position between characters can be split on or not
        max_segs_restricted = 2**(chars-n_pretokens)  # for every extra pretoken, 1 split position is fixed
        if self._do_logarithmic_segmentations:
            if segs == 0:
                print("Word", raw_word, "has no segmentations, so its log is -infinite. Skipping it.")
                return
            segs                = log2(segs)
            max_segs            = log2(max_segs)
            max_segs_restricted = log2(max_segs_restricted)

        # For metrics that compare against word length, it is possible to work in word space and not pretoken space.
        if self._do_measure_original_word_length:
            chars = len(raw_word)
        seg_to_char_ratio   = segs/chars
        seg_to_maxseg_ratio = segs/max_segs_restricted if chars-n_pretokens != 0 else 1

        # Accumulate
        self.sum_one          += 1*II
        self.sum_one_weighted += 1*f_w
        self.sum_seg          += segs*II
        self.sum_seg_weighted += segs*f_w

        self.sum_len          += chars*II
        self.sum_len_weighted += chars*f_w
        self.sum_seg_on_len          += seg_to_char_ratio*II
        self.sum_seg_on_len_weighted += seg_to_char_ratio*f_w

        self.sum_maxseg          += max_segs_restricted*II
        self.sum_maxseg_weighted += max_segs_restricted*f_w
        self.sum_seg_on_maxseg          += seg_to_maxseg_ratio*II
        self.sum_seg_on_maxseg_weighted += seg_to_maxseg_ratio*f_w
        # print(raw_word, "\t\t", segs, "of", max_segs, "or", max_segs_restricted)

    def _compute(self) -> VocabularyFertility:
        return VocabularyFertility(
            vocab_size=len(self.vocab),

            segmentations_per_word_type =self.sum_seg         /(self.sum_one or 1),
            segmentations_per_word_token=self.sum_seg_weighted/self.sum_one_weighted,
    
            segmentations_per_type_char_micro =self.sum_seg                /(self.sum_len or 1),
            segmentations_per_token_char_micro=self.sum_seg_weighted       /self.sum_len_weighted,
            segmentations_per_type_char_macro =self.sum_seg_on_len         /(self.sum_one or 1),
            segmentations_per_token_char_macro=self.sum_seg_on_len_weighted/self.sum_one_weighted,
    
            segmentations_per_type_max_micro =self.sum_seg                   /(self.sum_maxseg or 1),
            segmentations_per_token_max_micro=self.sum_seg_weighted          /self.sum_maxseg_weighted,
            segmentations_per_type_max_macro =self.sum_seg_on_maxseg         /(self.sum_one or 1),
            segmentations_per_token_max_macro=self.sum_seg_on_maxseg_weighted/self.sum_one_weighted
        )


class SegmentationProperties(FinallyObservableObserver[Tokens,InferenceFertility]):
    """
    Apply tokeniser segmentation and compute averages across words generated by pretokenising examples.
    """

    def __init__(self, track_unique_words: bool, observers: List[Observer[InferenceFertility]]):
        """
        :param track_unique_words: if True, will output an extra set of metrics wherein every unique word has equal
                                   contribution. This requires storing a set of all seen words, which consumes a lot of memory.
        """
        super().__init__(observers=observers)
        self._unique_words = track_unique_words

    def _initialiseAsObserver(self, identifier: str):
        self.seen = set()

        self.sum_one          = 0
        self.sum_one_weighted = 0
        self.sum_tk          = 0
        self.sum_tk_weighted = 0  # Equivalent to corpus token count (CTC).

        self.sum_len          = 0
        self.sum_len_weighted = 0  # Equivalent to corpus character count (CCC).
        self.sum_len_on_tk          = 0
        self.sum_len_on_tk_weighted = 0

        self.sum_tk_on_len          = 0
        self.sum_tk_on_len_weighted = 0
        self.sum_tk1_on_len1          = 0
        self.sum_tk1_on_len1_weighted = 0

    def _receive(self, sample: Tokens, weight: float):
        tokens, f_w = sample, weight
        if len(tokens) == 0:
            return

        II = 0
        if self._unique_words:
            word = "".join(tokens)
            if word not in self.seen:
                self.seen.add(word)
                II = 1

        chars = sum(map(len, tokens))
        tk    = len(tokens)
        char_to_token_ratio = chars/tk
        token_to_char_ratio = tk/chars
        segmentality        = (tk-1)/(chars-1) if chars > 1 else 1.0

        # Accumulate
        self.sum_one          += 1*II
        self.sum_one_weighted += 1*f_w
        self.sum_tk          += tk*II
        self.sum_tk_weighted += tk*f_w
        self.sum_len          += chars*II
        self.sum_len_weighted += chars*f_w
        self.sum_len_on_tk          += char_to_token_ratio*II
        self.sum_len_on_tk_weighted += char_to_token_ratio*f_w
        self.sum_tk_on_len          += token_to_char_ratio*II
        self.sum_tk_on_len_weighted += token_to_char_ratio*f_w
        self.sum_tk1_on_len1          += segmentality*II
        self.sum_tk1_on_len1_weighted += segmentality*f_w

    def _compute(self) -> InferenceFertility:
        return InferenceFertility(
            ccc=self.sum_len_weighted,
            ctc=self.sum_tk_weighted,
            cwc=self.sum_one_weighted,

            lcc=self.sum_len,
            ltc=self.sum_tk,
            lwc=self.sum_one,

            tokens_per_word_type=self.sum_tk/(self.sum_one or 1),
            tokens_per_word_token=self.sum_tk_weighted/self.sum_one_weighted,

            chars_per_word_type_token_micro=self.sum_len/(self.sum_tk or 1),
            chars_per_word_token_token_micro=self.sum_len_weighted/self.sum_tk_weighted,

            chars_per_word_type_token_macro=self.sum_len_on_tk/(self.sum_one or 1),
            tokens_per_word_type_char_macro=self.sum_tk_on_len/(self.sum_one or 1),
            chars_per_word_token_token_macro=self.sum_len_on_tk_weighted/self.sum_one_weighted,
            tokens_per_word_token_char_macro=self.sum_tk_on_len_weighted/self.sum_one_weighted,

            segmentality_word_types=self.sum_tk1_on_len1/(self.sum_one or 1),
            segmentality_word_tokens=self.sum_tk1_on_len1_weighted/self.sum_one_weighted
        )

    def _cachePath(self, unambiguous_cache_identifier: str) -> Path:
        return TkTkTPaths.extend(TkTkTPaths.pathToEvaluations(), ["fertility", "inference"]) / (unambiguous_cache_identifier + ".json")

    def _cacheLoad(self, cache_path: Path) -> InferenceFertility:
        return jsonToDataclass(InferenceFertility, cache_path)

    def _cacheStore(self, cache_path: Path, result: InferenceFertility):
        dataclassToJson(result, cache_path)
