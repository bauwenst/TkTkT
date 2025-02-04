from typing import Dict, List, Union

from fiject import StreamingMultiHistogram, StreamingVariableGranularityHistogram, BinSpec, VariableGranularityHistogram, FIJECT_DEFAULTS, HistoBars, CacheMode

from ...interfaces.tokeniser import TokeniserWithFiniteTypeDomain, Tokeniser
from ...interfaces.factories import Deserialiser
from ...util.timing import timeit
from ...util.iterables import intercalate, streamProgress, allEqual
from ...util.combinatorics import getLOCKey
from ...util.types import NamedIterable


@timeit
def visualiseCharsVersusTokensRelationships(
    tokenisers: List[TokeniserWithFiniteTypeDomain],
    raw_words: NamedIterable[str], counts: Dict[str, float]=None,
    n_samples_per_word: int=1, do_progressbar: bool=False,
    do_measure_original_word_length: bool=False, exclude_words_over_length: int=100
):
    """
    Produces the following histograms:
        1. CPT ratio (one per word): for each word: tokenise, compute len(word)/len(tokens) == 1/len(tokens) * \sum_{token} len(token), and add as a point to the histogram. This is the average amount of characters per token.
        2. Normalised fertility, a.k.a. segmentality (one per word): for each word: tokenise, compute len(tokens)-1/len(word)-1, and add it as a span to the histogram with total area 1/len(word) (spread over many bins).
        3. Length (multiple per word): for each word: tokenise, then for each token, add len(token) as a point to the histogram.
        4. Length (one per vocabulary type): for each type in the vocabulary: add len(type) as a point to the histogram.

    You would think the second one is just an incomprehensible version of the first, but this is not true: tokens/char
    is limited to ]0,1] (being able to approach 0 arbitrarily closely with bigger tokens) whilst chars/token is not limited
    and can be anywhere in [1, +infty[. Both have their use:
        - If the tokeniser is capable of representing words in full regardless of their length, histogram (1) will look
          like the length distribution of the corpus, whilst histogram (2) will pile up towards the left.
        - If the tokeniser is an N-gram model that always produces the same token length, histogram (1) will be peaked
          at N whilst histogram (2) will pile up around 1/N.

    You could renormalise graph (1) and (3) by the distribution of length in the vocabulary, but this only makes sense
    if we assume that all types in the vocabulary appear equally frequently (so if there are twice as many types of
    length 2 as there are of length 3, you would always see 2 dominate 3 since there are more of them). This assumption
    is completely wrong (tokens are used Zipfianly) and hence there's little use for such renormalisation.

    TODO: Maybe add a vocab id vs. frequency plot.

    :return: Summary statistics about the histogram on segmentality and the histogram on token lengths.
    """
    if not tokenisers:
        raise ValueError("No tokenisers given.")
    if counts is None:
        counts = dict()

    FIJECT_DEFAULTS.GLOBAL_STEM_PREFIX = (tokenisers[0].getName() if len(tokenisers) == 1 else "chars-vs-tokens") + "_" + raw_words.name

    histo_cpt_ratio           = StreamingMultiHistogram("cpt-ratios",                    binspec=BinSpec.halfopen(minimum=1, width=0.25))
    histo_segmentality        = StreamingVariableGranularityHistogram("tokens-per-char", binspec=BinSpec.closedFromAmount(minimum=0, maximum=1, amount=20))
    histo_chars_across_tokens = StreamingMultiHistogram("chars-in-tokens",               binspec=BinSpec.halfopen(minimum=1, width=1))
    histo_chars_across_types  = StreamingMultiHistogram("chars-in-types",                binspec=BinSpec.halfopen(minimum=1, width=1))

    if histo_chars_across_types.needs_computation:
        for tokeniser in tokenisers:
            name = tokeniser.getName()
            for t in tokeniser.types():
                histo_chars_across_types.add(name, len(t))

    if histo_cpt_ratio.needs_computation or \
       histo_segmentality.needs_computation or \
       histo_chars_across_tokens.needs_computation:
        for raw_word in streamProgress(raw_words) if do_progressbar else raw_words:
            n_raw_chars = len(raw_word)
            if n_raw_chars > exclude_words_over_length:
                continue

            for tokeniser in tokenisers:
                name = tokeniser.getName()
                for _ in range(n_samples_per_word):
                    # Compute metrics
                    tokens = tokeniser.prepareAndTokenise(raw_word)
                    n_token_chars = sum(map(len, tokens))

                    n_chars = n_raw_chars if do_measure_original_word_length else n_token_chars
                    n_tokens = len(tokens)
                    char_to_token_ratio = n_chars/n_tokens

                    f_w = counts.get(raw_word, 1)  # TODO: Frequency is unsupported by Fiject, currently.

                    # Add to histograms
                    histo_cpt_ratio.add(name, char_to_token_ratio, weight=f_w)
                    histo_segmentality.add(n_tokens-1, n_token_chars, weight=f_w, class_name=name)  # You can have 1 ... n_chars tokens. The interface requires the first argument to be in 0 ... n-1 (which makes sense, otherwise the first bin would never be added to and the first bin changes size depending on n_chars).
                    for token in tokens:
                        histo_chars_across_tokens.add(name, len(token), weight=f_w)

    histo_cpt_ratio.commit(StreamingMultiHistogram.ArgsGlobal(
        x_label="Characters-per-token ratio $R$",
        x_center_ticks=False,
        x_tickspacing=0.25,
        y_label="Fraction of words",
        relative_counts=True
    ))
    histo_segmentality.commit(StreamingVariableGranularityHistogram.ArgsGlobal(
        x_label=r"Segmentality $\mathcal S$ (word tokens $\to$ character tokens)",  # Lowest is 1/n, highest is 1/1. The value 1/n adds to all bins between 0 and 1/(n+1), lumping it in with a ratio of 0 which is impossible.
        y_label="Fraction of words",
        relative_counts=True,
        x_tickspacing=0.1
    ))
    histo_chars_across_tokens.commit(StreamingMultiHistogram.ArgsGlobal(
        x_label="Characters",
        y_label="Fraction of tokens",
        relative_counts=True,
        x_center_ticks=True,
        x_tickspacing=1
    ))
    histo_chars_across_types.commit(StreamingMultiHistogram.ArgsGlobal(
        x_label="Characters",
        y_label="Fraction of vocabulary types",
        relative_counts=True,
        x_center_ticks=True,
        x_tickspacing=1
    ))

    # Also commit the token histogram normalised by the type histogram (lengths that are 2x as popular in the vocab should have their frequency /2 assuming tokens are randomly drawn from the vocab).
    histo_chars_across_tokens_norm = StreamingMultiHistogram(histo_chars_across_tokens.name + "_no-prior", binspec=histo_chars_across_tokens.bins)
    histo_chars_across_tokens_norm.data = histo_chars_across_tokens.data.copy()
    histo_chars_across_tokens_norm.commit(StreamingMultiHistogram.ArgsGlobal(
        x_label="Characters",
        y_label="Fraction of tokens (after dividing token frequencies by their vocabulary prior)",
        relative_counts=True,
        x_center_ticks=True,
        x_tickspacing=1
    ), bin_reweighting={
        tokeniser.getName(): {k: 1/v for k,v in histo_chars_across_types.data[tokeniser.getName()].items()}  # Note: these counts are indexed by bin, and since the bins of both histos match, you don't have to shift k etc.
        for tokeniser in tokenisers
    })

    FIJECT_DEFAULTS.GLOBAL_STEM_PREFIX = ""

    # Summary statistics
    return histo_segmentality.getSummaries(), histo_chars_across_tokens.getSummaries().popitem()[1]


def visualiseTypes(vocabulary_sources: List[Union[TokeniserWithFiniteTypeDomain, Deserialiser]], names: List[str]=None):
    if not vocabulary_sources:
        raise ValueError("No vocabularies given.")

    if not names:
        names = [(source.getName() if isinstance(source, Tokeniser) else source.__class__.__name__) for source in vocabulary_sources]

    FIJECT_DEFAULTS.GLOBAL_STEM_PREFIX = names[0]*(len(vocabulary_sources) == 1)

    histo_chars_across_types = StreamingMultiHistogram("chars-in-types", binspec=BinSpec.halfopen(minimum=1, width=1), overwriting=True)
    for name,source in zip(names,vocabulary_sources):
        if isinstance(source, Tokeniser):
            type_iterator = source.types()
        elif isinstance(source, Deserialiser):
            type_iterator = source.buildVocabulary().keys()
        else:
            raise TypeError

        for t in type_iterator:
            histo_chars_across_types.add(name, len(t))

    histo_chars_across_types.commit(StreamingMultiHistogram.ArgsGlobal(
        x_label="Characters",
        x_tickspacing=1,
        x_lims=(0.25,20.75),
        x_center_ticks=True,

        y_label="Fraction of vocabulary types",
        y_tickspacing=1,
        relative_counts=True,

        # aspect_ratio=(4*1.25,3*1.25)
    ))

    FIJECT_DEFAULTS.GLOBAL_STEM_PREFIX = ""


def visualiseSingleWordSegmentationDistribution(tokenisers: List[Tokeniser], word: str, samples: int=1_000_000,
                                                segmentation_histogram_max_bins: int=2**16, do_bitbased_ordering: bool=False):
    """
    Produces the following histograms:
        1. For a given word w, on the x-axis all 2^{|w|-1} segmentations, and on the y-axis the amount of times it was produced by the tokeniser.
           This represents one big distribution.
        2. For a given word, on the x-axis the |w|-1 possible split points, and on the y-axis the amount of times a split is put there.
           This represents |w|-1 Bernoulli distributions.
        3. For a given word, on the x-axis all |w| possible amounts of tokens, and on the y-axis the amount of times a segmentation has that amount.
           In other words, the sum of the above Bernoulli variables, and hence a binomial distribution.
           Has the same shape as the tokens/chars distribution except on {1, ..., |w|} rather than [0,1].
           TODO: You could reweight this by a binomial distribution that assumes p = 0.5, proportional to |w|-choose-x, to correct
                 for the fact that e.g. |w|/2 should dominate because there are more ways to form |w|/2 tokens from |w| characters
                 than any other amount of tokens. Or, instead of reweighting, you could turn it into a Q-Q plot :)
        4. For a given word, on the x-axis all |w| possible lengths of tokens, and on the y-axis the amount of times
           such a token is produced across all experiments.
        5. For a given word, the distribution of characters-per-token ratios across all samples. Has the same shape as
           the 1/tokens distribution except stretched by n. 1/tokens has range [1/n, 1/1] which is limited to [0,1] but
           is impossible to compare between words since you need finer and finer resolution bins to show what's going on near 0 for larger words.
           chars/tokens has range [n/n,n/1] == [1,n] which can grow to infinity and doesn't need resolution changes.

    We only do one word at a time because especially histogram (1) cannot be combined across multiple words since the
    distribution of lengths across segmentations (both token lengths and token amounts) is not proportioned the same
    for different lengths of words.

    ---

    The ordering on the x-axis of histogram (1) is quite tricky. What we want to show with our graphs is that there is
    a skew towards small tokens. This concept is hard to define formally: if you order the segmentations by amount of tokens,
    that doesn't actually put small tokens on one side of the axis: you can have 3 tokens where 2 are single characters.
    Ordering by the chars/token ratio is the same as ordering by 1/tokens since all segmentations have the same characters,
    and 1/tokens is just the mirror image of ordering by the amount of tokens. Also, 5+1 characters and 3+3 characters have
    the same chars/token ratio.

    The most popular amount of tokens is that in the middle of the axis, because it has the most segmentations that fall into
    it (namely n-choose-n/2). When segmentations are spread uniformly across the axis and then you apply binning based on
    amount of tokens, a bell curve will appear. If you see bad segmentations most of the time, then that means the amounts of
    tokens around this bell curve's centre lobe produce bad segmentations.
    One transformation you could apply to the axis is hence ordering from "normie" amounts to rare amounts.

    Within one amount, you could sort the segmentations by the key you get as follows: take a segmentation's token lengths
    as a tuple and sort them from low to high. Hence, (1,1,2,3) comes before (1,2,2,2).

    An alternative order is doing the above with ALL such tuples, left-aligned. The difference is that a segmentation like 1+9
    is on the few-token side in one order, and at the end of the 1-starting bin in the other order.
    """
    if not tokenisers:
        raise ValueError("No tokenisers given.")

    FIJECT_DEFAULTS.GLOBAL_STEM_PREFIX = (tokenisers[0].getName() if len(tokenisers) == 1 else "chars-vs-tokens") + f"_{word}×{samples}"

    assert allEqual(sum(map(len, tokeniser.preprocessor.do(word))) for tokeniser in tokenisers)
    n_chars = sum(map(len, tokenisers[0].preprocessor.do(word)))

    histo_across_segmentations        = StreamingMultiHistogram("segmentations" + ("-bb" if do_bitbased_ordering else "-loc"), BinSpec.closedFromAmount(minimum=0, maximum=2**(n_chars-1), amount=min(2**(n_chars-1),segmentation_histogram_max_bins)), caching=CacheMode.IF_MISSING)
    histo_across_amounts              = StreamingMultiHistogram("amounts", BinSpec.closedFromAmount(minimum=1, maximum=n_chars+1, amount=n_chars), caching=CacheMode.IF_MISSING)
    bars_foreach_splitpoint           = HistoBars("split-heatmap", caching=CacheMode.IF_MISSING)
    histo_across_token_lengths        = StreamingMultiHistogram("lengths", BinSpec.closedFromAmount(minimum=1, maximum=n_chars+1, amount=n_chars), caching=CacheMode.IF_MISSING)
    histo_across_char_per_token_ratio = StreamingMultiHistogram("cpt-ratio", BinSpec.halfopen(minimum=1, width=0.25), caching=CacheMode.IF_MISSING)

    if histo_across_segmentations.needs_computation or \
           histo_across_amounts.needs_computation or \
           bars_foreach_splitpoint.needs_computation or \
           histo_across_token_lengths.needs_computation or \
           histo_across_char_per_token_ratio.needs_computation:
        for tokeniser in tokenisers:
            name = tokeniser.getName()
            split_heatmap = [0] * (n_chars - 1)

            for _ in streamProgress(range(samples)):
                tokens = tokeniser.prepareAndTokenise(word)
                assert sum(map(len, tokens)) == n_chars  # Basically asserts that the tokeniser is non-degenerate.

                # Easy histograms
                amount = len(tokens)
                lengths = list(map(len, tokens))

                histo_across_amounts.add(name, amount)
                for l in lengths:
                    histo_across_token_lengths.add(name, l)

                histo_across_char_per_token_ratio.add(name, n_chars/amount)

                # Histograms that need a segmentation map
                bits = getSegmentationBitstring(tokens)

                if do_bitbased_ordering:
                    histo_across_segmentations.add(name, int(bits,2))
                else:
                    histo_across_segmentations.add(name, getLOCKey(lengths))

                for split_position,bit in enumerate(bits):
                    if bit == "1":
                        split_heatmap[split_position] += 1

            for split_count in split_heatmap:
                bars_foreach_splitpoint.append(name, split_count/samples)

    # Commit all the plots
    preprocessed_word = ''.join(tokenisers[0].preprocessor.do(word))

    histo_across_segmentations.commit(StreamingMultiHistogram.ArgsGlobal(
        x_label=f"Segmentation key ({'bit-based' if do_bitbased_ordering else 'LOC'})",

        y_label="Fraction of samples",
        relative_counts=True,
        log_y=False
    ))
    histo_across_segmentations.raw_name = histo_across_segmentations.raw_name + "_log"
    histo_across_segmentations.commit(StreamingMultiHistogram.ArgsGlobal(
        x_label=f"Segmentation key ({'bit-based' if do_bitbased_ordering else 'LOC'})",

        y_label="Fraction of samples",
        relative_counts=True,
        log_y=True,
        # y_lims=(1e-2,1e0)
        # y_lims=(1e-10, 1e2)
    ))
    histo_across_amounts.commit(StreamingMultiHistogram.ArgsGlobal(
        x_label="Amount of tokens",
        x_tickspacing=1,
        x_center_ticks=True,

        y_label=f"Fraction of samples",
        relative_counts=True
    ))
    histo_across_token_lengths.commit(StreamingMultiHistogram.ArgsGlobal(
        x_label="Token length",
        x_tickspacing=1,
        x_center_ticks=True,
        x_lims=(0.5,11.5),

        y_label="Fraction of tokens across samples",
        relative_counts=True
    ))
    bars_foreach_splitpoint.commit(HistoBars.ArgsGlobal(
        y_tickspacing=0.1,
        x_label=f"Split index in {preprocessed_word}",
        y_label="Fraction of samples where the index is split on",
        bar_width=0.9
    ))
    histo_across_char_per_token_ratio.commit(StreamingMultiHistogram.ArgsGlobal(
        x_label="Characters-per-token ratio",
        x_tickspacing=1,

        y_label="Fraction of words",
        relative_counts=True
    ))

    FIJECT_DEFAULTS.GLOBAL_STEM_PREFIX = ""


def getSegmentationMask(tokens: List[str]) -> List[int]:
    return sum(intercalate(map(lambda i: (i-1)*[0], map(len, tokens)), [1]), start=[])


def getSegmentationBitstring(tokens: List[str]) -> str:
    return "".join(intercalate(map(lambda i: (i-1)*"0", map(len, tokens)), "1"))
