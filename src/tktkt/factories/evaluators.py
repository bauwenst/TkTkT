"""
Example evaluation pipelines.
"""
from ..evaluation.observing import *
from ..evaluation.morphological import *
from ..evaluation.entropy import *
from ..interfaces import TokeniserWithVocabulary
from ..util.types import Number


def isIterableWithCounts(iterable: Union[Iterable[str], Iterable[tuple[str,Number]]]) -> bool:
    for thing in iterable:
        if isinstance(thing, tuple) and len(thing) == 2 and isinstance(thing[0], str) and isinstance(thing[1], (int,float)):
            return True
        else:
            return False
    return False


def evaluateTokeniser(experiment_id: str, corpus: Union[NamedIterable[str],NamedIterable[tuple[str,float]]], tokeniser: Tokeniser, token_consumers: list[Observer[Tokens]]):
    """
    Functional shorthand for one specific instance of the general object-oriented Observable/Observer approach,
    for cases where you want to tokenise strings in a corpus and compute metrics over the tokeniser's token outputs.
    """
    ObservableIterable(
        experiment_id=experiment_id,
        already_contains_weights=isIterableWithCounts(corpus),
        iterable=corpus,
        observers=[
            ObservableTokeniser(
                tokeniser=tokeniser,
                observers=token_consumers
            )
        ]
    ).run()


def evaluateTokeniserOnWords(experiment_id: str, corpus: Union[NamedIterable[str],NamedIterable[tuple[str,float]]], word_preprocessor: Preprocessor, tokeniser: Tokeniser, token_consumers: list[Observer[Tokens]]):
    """
    Same as evaluateTokeniser except now you split the strings into "words" beforehand.
    """
    ObservableIterable(
        experiment_id=experiment_id,
        already_contains_weights=isIterableWithCounts(corpus),
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


def evaluateTokeniserOnMorphology(experiment_id: str, dataset: ModestDataset, tokeniser: Tokeniser, effective_preprocessor: Preprocessor=None,
                                  has_freemorphsplit: bool=False, output: DataclassObserver=None):
    """
    Like evaluateTokeniserOnWords except with words with a morphological segmentation reference.
    Don't forget to fence after this.

    :param has_freemorphsplit: Whether the dataset provides a segmentation into non-bound morphemes.
    """
    if effective_preprocessor is None:
        effective_preprocessor = tokeniser.preprocessor

    connection = WirelessObserverConnection()
    if output is None:
        output = DataclassObserver()
    output.addMetadata({"corpus": dataset.identifier(), "name": tokeniser.getName()})

    MorphologyIterable(
        experiment_id=experiment_id,
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
                                observers=[
                                    MorphologyAsClassification(
                                        MorphSplit(),
                                        effective_preprocessor=effective_preprocessor,
                                        observers=[ConfusionMatrixSummary(observers=[output.withSuffix("morph")])]
                                    )
                                ] + has_freemorphsplit*[
                                    MorphologyAsClassification(
                                        FreeMorphSplit(),
                                        effective_preprocessor=effective_preprocessor,
                                        observers=[ConfusionMatrixSummary(observers=[output.withSuffix("free")])]
                                    )
                                ]
                            )
                        ]
                    )
                ]
            )
        ]
    ).run()
    return output


class ObserverWithTTRandEntropy(ObservableTokeniser):
    """Receives text, tokenises it, and runs all of the above metrics on the result. Prints the results."""
    def __init__(self, tokeniser: TokeniserWithVocabulary, renyi_alpha: float, mattr_window_size: int, mattr_stride: int, endpoint: Observer[Any]=PrintingObserver()):
        super().__init__(
            tokeniser=tokeniser,
            observers=[
                TokenUnigramDistribution(
                    ensured_vocabulary=tokeniser.types(),
                    observers=[
                        TTR(observers=[endpoint]),
                        RenyiEntropy(
                            alpha=renyi_alpha,
                            vocab_size_in_denominator=len(tokeniser.vocab),
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
