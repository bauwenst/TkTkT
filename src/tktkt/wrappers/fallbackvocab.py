from typing import Iterable

from ..interfaces.tokenisers import *
from ..util.iterables import foldSpans


class TokeniserWithByteFallback(TokeniserWithVocabulary[WithSpecials]):
    """
    The idea is that instead of producing an UNK, you still encode the given string losslessly, e.g. by turning it into
    Unicode bytes which are then added to the existing vocabulary.

    Not the same as a byte-based tokeniser, which FIRST applies a byte-based mapping and THEN tokenises. This one FIRST
    tokenises and THEN applies a byte-based mapping.
    """

    def __init__(self, tokeniser_with_unks: TokeniserWithVocabulary[WithSpecials], crude_fallback: bool=False):
        """
        :param crude_fallback: When a word contains an UNK, re-encode the entire thing with the fallback vocabulary,
                               rather than only re-encoding the part of the pretoken that actually produced the UNK.
                               Faster, but more fragile (one character can change the entire segmentation of the word).
        """
        assert tokeniser_with_unks.vocab.UNK is not None
        self.core = tokeniser_with_unks
        self.crude = crude_fallback

        self.byte_types = {i: f"[BYTE {i}]" for i in range(256)}
        for typ in self.byte_types.values():
            if typ in self.core.vocab:
                raise ValueError(f"Surrogate type already exists in core vocabulary: {typ}")

        self.unk = ""

        first_available_id = max(self.core.vocab.values()) + 1
        combined_vocabulary = self.core.vocab | {typ: first_available_id+i for i,typ in enumerate(self.byte_types.values())}
        super().__init__(preprocessor=self.core.preprocessor, vocab=Vocab(
            sorted(combined_vocabulary.keys(), key=combined_vocabulary.get),
            specials=self.core.vocab.specials,
            unk_id=self.core.vocab.UNK
        ))

        # Cache UNK ID (also a sanity check that there is an UNK, otherwise we can't use it to detect when we need fallback and need to do vocab checks ourselves)
        self.unk_id = self.typeToId(self.unk)

    def tokenise(self, pretoken: str) -> Tokens:
        tokens = self.core.tokenise(pretoken)

        # First detect if there are segments that are literally the UNK token, or segments that aren't recognised by the
        # vocab and need to be replaced by the UNK token.
        has_unk = False
        for i in range(len(tokens)):
            if self.typeToId(tokens[i]) == self.unk_id:  # Either tokens[i] == self.unk or it is any string not in the vocab.
                tokens[i] = self.unk
                has_unk = True

        if not has_unk:
            return tokens
        elif self.crude:
            return list(map(self.byte_types.get, pretoken.encode("utf-8")))
        else:
            tokens, are_unks = self.decodeUnks(pretoken, tokens)
            final_tokens = []
            for token, was_unk in zip(tokens, are_unks):
                if was_unk:
                    final_tokens.extend(map(self.byte_types.get, token.encode("utf-8")))
                else:
                    final_tokens.append(token)
            return final_tokens

    def decodeUnks(self, pretoken: str, tokens: list[str]) -> tuple[Iterable[str], Iterable[bool]]:
        tokens = list(foldSpans(tokens, self.unk))

        # Align tokens with string to find out where the UNK(s) are.
        # - Viterbi for alignment: you have N characters to traverse, and a list of T-U normal tokens and U [UNK] tokens.
        #   You want to consume all T tokens. A non-UNK is consumed by matching its characters. An UNK is consumed by
        #   skipping characters. All tokens must be consumed by the end.
        # - In unresolvably ambiguous cases, e.g. "prefixaaabcccbdddsuffix" -> [prefix, UNK, b, UNK, suffix] (which would
        #   not happen in a sensible tokeniser, because "b" would be recognised twice), where the UNKs could be either
        #   "aaabccc"/"ddd" or they could be "aaa"/"cccbddd", the heuristic is applied that the sequence that has a
        #   history of mapping shorter strings to UNKs is preferred. In this case, the second one, because when it
        #   arrives at the second UNK, it has had to resort to UNKs minimally.
        trellis = [[None for _ in range(len(pretoken)+1)] for _ in range(len(tokens)+1)]  # trellis[i][j] means i tokens consumed and j characters consumed.
        trellis[0][0] = 0
        for i in range(len(pretoken)):
            # You have consumed i (column) characters. Now you want to step.
            for j in range(len(tokens)):
                # You have consumed j (row) tokens. The next token is token j.
                if trellis[j][i] is None:  # Impossible to get here.
                    continue

                step = tokens[j]
                if step != self.unk:
                    if pretoken[i:i+len(step)] == step:  # You step exactly that many characters and that's it.
                        trellis[j+1][i+len(step)] = i
                else:  # You get to skip any amount (> 0) of characters you want.
                    for k in range(i+1,len(pretoken)+1):
                        if trellis[j+1][k] is not None:  # If you're not the first to get there, an earlier column (lower index i) already reached this UNK and gets priority for all subsequent strings.
                            break
                        trellis[j+1][k] = i

        # Backtrace
        current_col = len(pretoken)
        current_row = len(tokens)

        actual_tokens = []
        unk_map = []
        while current_row > 0:
            next_col = trellis[current_row][current_col]

            actual_tokens.append(pretoken[next_col:current_col])
            unk_map.append(tokens[current_row-1] == self.unk)  # The token consumed to get here.

            current_col = next_col
            current_row -= 1

        return reversed(actual_tokens), reversed(unk_map)
