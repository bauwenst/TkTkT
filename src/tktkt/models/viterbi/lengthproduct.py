from typing import List

from ...interfaces.general import TokeniserWithVocab


class RA_Product(TokeniserWithVocab):
    """
    The idea: we want to nicely distribute subwords over the string. One property of multiplication with a fixed sum is
    that it is highest when all numbers are as close as possible.
    Hence, this finds the segmentation with maximal product of subword lengths.
    TODO: Okay, in hindsight, it works, but it's too balanced. You might want to incorporate something into the
        score that nudges it towards fewer substrings.
        For a string of 6, it is true that
            1* 5*1 < 1* 6 = 1* 3*2*1 < 1* 4*2 = 1* 2*2*2 < 1* 3*3
        So the ordering of a product is... weird.
    """

    def tokenise(self, word: str) -> List[str]:
        scores = [0 for _ in range(len(word) + 1)]
        scores[0] = 1
        backpointer = [None for _ in range(len(word) + 1)]

        # Forward pass
        for start in range(len(word)):
            for end in range(start + 1, len(word) + 1):
                step = word[start:end]
                new_score = scores[start] * len(step)
                if step in self.vocab and new_score > scores[end]:
                    scores[end] = scores[start] * len(step)
                    backpointer[end] = start

        # Backward pass
        tokens = []
        current_idx = len(word) + 1
        next_idx = backpointer[-1]
        while next_idx is not None:
            tokens.append(word[next_idx:current_idx])
            current_idx = next_idx
            next_idx = backpointer[next_idx]
        tokens.reverse()
        return tokens
