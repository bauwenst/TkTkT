"""
Implementation of TreeTok inference. I.e.: you have a list of probabilities for having a split point between each
character, and you check if the current string is in the vocabulary, where you stop if yes and you do the same thing
on the halves separated by the largest probability if no.
https://aclanthology.org/2025.findings-acl.1146.pdf
"""
from .viterbi import CharacterClassifier
from ...interfaces.tokenisers import *
from ...util.iterables import maxargmax


class TopDownTree(TokeniserWithVocabulary[WithSpecials]):

    def __init__(self, preprocessor_before_classifier: Preprocessor, preprocessor_after_classifier: Preprocessor,
                 vocab: Vocab[WithSpecials]):
        super().__init__(preprocessor=preprocessor_before_classifier, vocab=vocab)
        self.hard_boundaries = preprocessor_after_classifier

    def setClassifier(self, cls: CharacterClassifier):
        self._classifier = cls

    def tokenise(self, pretoken: str) -> Tokens:
        if pretoken in self.vocab:
            return [pretoken]

        tokens = []
        probabilities = self._classifier.getPointLogProbabilities(pretoken)
        start = 0
        for pretoken in self.hard_boundaries.do(pretoken):
            length = len(pretoken)
            tokens.extend(self._tokeniseRecursively(pretoken, probabilities[start:start+length]))
            start += length

        return tokens

    def _tokeniseRecursively(self, string: str, probabilities: list[float]) -> list[str]:
        if string in self.vocab:
            return [string]

        _, index = maxargmax(probabilities[:-1])
        return self._tokeniseRecursively(string[:index+1], probabilities[:index+1]) + self._tokeniseRecursively(string[index+1:], probabilities[index+1:])



# TODO: There is probably some similar top-down approach without probabilities, where you just try to minimise the
#       depth of the tree or something like this. Needs DP obviously.