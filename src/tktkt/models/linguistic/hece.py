"""
Based on the code at https://github.com/senolgulgonul/hecetokenizer/blob/main/hecetokenizer.py

Gulgonul - HeceTokenizer: A Syllable-Based Tokenization Approach for Turkish Retrieval
https://arxiv.org/abs/2604.10665

TODO: Should support space markers, capitals, and non-alphabetic inputs.
"""
from ...interfaces.tokenisers import Tokeniser, Tokens


TURKISH_VOWELS = set("aeıioöuü")


def is_vowel(c):
    return c in TURKISH_VOWELS


def is_consonant(c):
    return c not in TURKISH_VOWELS


class Hece(Tokeniser):

    def tokenise(self, pretoken: str) -> Tokens:
        """
        Syllabify a single Turkish word using right-to-left greedy pattern matching.

        Six canonical syllable patterns (V=vowel, C=consonant):
            CVCC, VCC, CVC, VC, CV, V

        Isolated consonants (e.g. from loanword clusters) are handled in step g.

        Returns:
            list[str]: List of syllables.
        """
        assert not any(c.isupper() or c.isnumeric() for c in pretoken)

        if not pretoken:
            return []

        syllables = []
        i = len(pretoken) - 1

        while i >= 0:
            n_remaining = i + 1

            # a. CVCC
            if n_remaining >= 4 and is_consonant(pretoken[i - 3]) and is_vowel(pretoken[i - 2]) and is_consonant(pretoken[i - 1]) and is_consonant(pretoken[i]):
                syllables.insert(0, pretoken[i - 3:i + 1])
                i -= 4

            # b. VCC
            elif n_remaining >= 3 and is_vowel(pretoken[i - 2]) and is_consonant(pretoken[i - 1]) and is_consonant(pretoken[i]):
                syllables.insert(0, pretoken[i - 2:i + 1])
                i -= 3

            # c. CVC
            elif n_remaining >= 3 and is_consonant(pretoken[i - 2]) and is_vowel(pretoken[i - 1]) and is_consonant(pretoken[i]):
                syllables.insert(0, pretoken[i - 2:i + 1])
                i -= 3

            # d. VC
            elif n_remaining >= 2 and is_vowel(pretoken[i - 1]) and is_consonant(pretoken[i]):
                syllables.insert(0, pretoken[i - 1:i + 1])
                i -= 2

            # e. CV
            elif n_remaining >= 2 and is_consonant(pretoken[i - 1]) and is_vowel(pretoken[i]):
                syllables.insert(0, pretoken[i - 1:i + 1])
                i -= 2

            # f. V
            elif is_vowel(pretoken[i]):
                syllables.insert(0, pretoken[i])
                i -= 1

            # g. isolated consonant (loanword clusters)
            else:
                syllables.insert(0, pretoken[i])
                i -= 1

        return syllables
