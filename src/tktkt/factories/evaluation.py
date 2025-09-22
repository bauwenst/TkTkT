"""
Example evaluation pipelines.
"""
from ..evaluation.observing import *
from ..evaluation.morphological import *
from ..evaluation.entropy import *


def evaluateTokeniser(corpus: NamedIterable[str], tokeniser: Tokeniser, token_consumers: List[Observer[Tokens]]):
    """
    Functional shorthand for one specific instance of the general object-oriented Observable/Observer approach,
    for cases where you want to tokenise strings in a corpus and compute metrics over the tokeniser's token outputs.
    """
    ObservableIterable(
        iterable=corpus,
        observers=[
            ObservableTokeniser(
                tokeniser=tokeniser,
                observers=token_consumers
            )
        ]
    ).run()


def evaluateTokeniserOnWords(corpus: NamedIterable[str], word_preprocessor: Preprocessor, tokeniser: Tokeniser, token_consumers: List[Observer[Tokens]]):
    """
    Same as evaluateTokeniser except now you split the strings into "words" beforehand.
    """
    ObservableIterable(
        iterable=corpus,
        observers=[
            ObservablePreprocessor(
                preprocessor=word_preprocessor,
                observers=[
                    ObservableTokeniser(
                        tokeniser=tokeniser,
                        observers=token_consumers
                    )
                ]
            )
        ]
    ).run()


def evaluateTokeniserOnMorphology(dataset: ModestDataset, tokeniser: Tokeniser, has_freemorphsplit: bool=False):
    """
    Like evaluateTokeniserOnWords except with words with a morphological segmentation reference.
    The downstream observers have already been specified for you.

    :param has_freemorphsplit: Whether the dataset provides a segmentation into non-bound morphemes.
    """
    connection = WirelessObserverConnection()
    results = DataclassCollectorObserver()
    results.addMetadata({"corpus": dataset.identifier(), "name": tokeniser.getName()})

    MorphologyIterable(
        dataset=dataset,
        observers=[
            WirelessSplittingObserver(  # Retains morphology objects until wireless receiver asks for it back.
                connection=connection,
                observers=[
                    ObservableTokeniser(
                        tokeniser=tokeniser,
                        observers=[
                            WirelessRecombiningObserver(
                                connection=connection,  # Asks for morphology object back.
                                observers=
                                [MorphologyAsClassification(MorphSplit(),     observers=[results.withSuffix("morph")])] +
                                [MorphologyAsClassification(FreeMorphSplit(), observers=[results.withSuffix("free" )])]*has_freemorphsplit
                            )
                        ]
                    )
                ]
            )
        ]
    ).run()
    return results.assemble()[0]  # TODO: You should allow the user to put this in a loop as for the other functions. .assemble() is only really appropriate when the user does it.


class ObserverWithTTRandEntropy(ObservableTokeniser):
    """Receives text, tokenises it, and runs all of the above metrics on the result. Prints the results."""
    def __init__(self, tokeniser: TokeniserWithFiniteTypeDomain, renyi_alpha: float, mattr_window_size: int, mattr_stride: int, endpoint: Observer[Any]=PrintingObserver()):
        super().__init__(
            tokeniser=tokeniser,
            observers=[
                TokenUnigramDistribution(
                    ensured_vocabulary=tokeniser.types(),
                    observers=[
                        TTR(observers=[endpoint]),
                        RenyiEntropy(
                            alpha=renyi_alpha,
                            vocab_size_in_denominator=tokeniser.getVocabSize(),
                            observers=[endpoint]
                        )
                    ]
                ),
                MATTR(
                    window_size=mattr_window_size,
                    stride=mattr_stride,
                    flush_every_example=False,
                    observers=[endpoint]
                )
            ],
        )
