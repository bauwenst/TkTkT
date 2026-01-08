"""
Greedy tokenisers that choose their tokens by starting at one edge of the input and moving to the other edge.

According to HuggingFace, L2R_Greedy is the algorithm used by WordPiece for inference. This is not true for the
original WordPiece paper, but it is true for the BERT paper.

I add a right-to-left version and lazy variants of both (explained below). Also a left-to-right merge-based tokeniser.

Implementations adapted from my master's thesis (Bauwens, 2023). https://bauwenst.github.io/cdn/doc/pdf/2023/masterthesis.pdf
"""
from math import inf

from ...interfaces.tokenisers import *


class L2R_Greedy(TokeniserWithVocabulary[WithSpecials]):
    """
    Note that you can write a left-to-right tokeniser in two ways:
        - Lempel-Ziv-style, where you select progressively larger subwords and stop the moment you encounter a subword
          not in the vocab;
        - The inverse of that, i.e. you select progressively smaller subwords and stop the moment you encounter a subword
          that is in the vocab;

    This class implements the latter. It is "greedy" like regex operators by default: make the match as big as possible,
    even if that means you pass the first match to go look for longer matches while meanwhile passing over strings that don't match.
    """

    def tokenise(self, word: str) -> Tokens:
        tokens = []
        token_length = len(word)
        while word:
            if token_length == 0 or word[:token_length] in self.vocab:
                if token_length == 0:  # This means the previous iteration, 'word[:1] in self.vocab' was False. Hence, there are no tokens for the remaining word. Pretend like the character does belong to the vocab. (Will be UNK'ed as a type.) TODO: Alternative is to produce self.unk.
                    token_length = 1

                tokens.append(word[:token_length])
                word = word[token_length:]
                token_length = len(word)
            else:
                token_length -= 1

        return tokens

MaxMatch = L2R_Greedy


class L2R_Lazy(TokeniserWithVocabulary[WithSpecials]):
    """
    "Lazy" like a lazy regex operator: matches as few characters as possible in order to have a match, i.e. find a string
    unknown to the vocabulary, at which point the token so far will be grouped.

    In the edge case that a single-character token is already unknown to the vocabulary, it is segmented off, but will
    be mapped to an UNK ID when looked up in the vocabulary.
    (The alternative, i.e. keep looking for a 2-character, 3-character, ... token that is known and then stopping when
    you again find a string that is unknown, probably won't work well, since that single character probably also won't
    exist inside bigger types.)
    """

    def tokenise(self, word: str) -> Tokens:
        tokens = []

        token_start  = 0
        token_length = 0
        while token_start < len(word):
            # Make a token if extending the length of the current token by 1 (looking ahead one position) gives an unknown token, or reaches past the end of the string (i.e. so far the token is in the vocab, and there are no more characters to add).
            if token_start+token_length == len(word) or word[token_start:token_start+token_length+1] not in self.vocab:
                if token_length == 0:  # This means you don't even recognise the current character. Pretend like you recognised the isolated character as a token. TODO: Alternatively, append self.unk.
                    token_length = 1

                tokens.append(word[token_start:token_start+token_length])
                token_start += token_length
                token_length = 0
            else:
                token_length += 1

        return tokens


class R2L_Greedy(TokeniserWithVocabulary[WithSpecials]):

    def tokenise(self, word: str) -> Tokens:
        tokens = []
        token_length = len(word)
        while word:
            if token_length == 0 or word[len(word)-token_length:] in self.vocab:
                if token_length == 0:
                    token_length = 1
                tokens.append(word[len(word)-token_length:])
                word = word[:len(word)-token_length]
                token_length = len(word)
            else:
                token_length -= 1

        tokens.reverse()
        return tokens


class R2L_Lazy(TokeniserWithVocabulary[WithSpecials]):

    def tokenise(self, word: str) -> Tokens:
        tokens = []

        token_end    = len(word)  # exclusive
        token_length = 0
        while token_end > 0:
            if token_end-token_length == 0 or word[token_end-token_length-1:token_end] not in self.vocab:
                if token_length == 0:
                    token_length = 1

                tokens.append(word[token_end-token_length:token_end])
                token_end -= token_length
                token_length = 0
            else:
                token_length += 1

        tokens.reverse()
        return tokens


class L2R_R2L_Alternating(TokeniserWithVocabulary[WithSpecials]):

    def __init__(self, preprocessor: Preprocessor, vocab: Vocab[WithSpecials],
                 start_left: bool=True):
        super().__init__(preprocessor=preprocessor, vocab=vocab)
        self._start_left = start_left

    def tokenise(self, pretoken: str) -> Tokens:
        left = self._start_left
        left_cursor  = 0  # Tokens from the left start on this cursor.
        right_cursor = len(pretoken)-1  # Tokens from the right end on this cursor. (inclusive indexing)

        left_tokens  = []
        right_tokens = []

        while left_cursor <= right_cursor:
            if left:
                for end in range(right_cursor, left_cursor-1, -1):
                    token = pretoken[left_cursor:end+1]
                    if token in self.vocab:
                        left_cursor = end+1
                        left_tokens.append(token)
                        break
                else:
                    raise ValueError(f"Character not in vocab: {pretoken[left_cursor]}")
            else:  # Yes, you could turn these two loops into one loop using some kind of subtraction. No, I'm not going to bother spaghettifying my code.
                for start in range(left_cursor, right_cursor+1):
                    token = pretoken[start:right_cursor+1]
                    if token in self.vocab:
                        right_cursor = start-1
                        right_tokens.insert(0, token)
                        break
                else:
                    raise ValueError(f"Character not in vocab: {pretoken[right_cursor]}")

            left = not left

        return left_tokens + right_tokens


class L2R2L_Greedy(TokeniserWithVocabulary[WithSpecials]):

    def __init__(self, preprocessor: Preprocessor, vocab: Vocab[WithSpecials],
                 prefer_left: bool=True):
        super().__init__(preprocessor=preprocessor, vocab=vocab)
        self._prefer_left = prefer_left

    def tokenise(self, pretoken: str) -> Tokens:
        # Inclusive indices
        left_start = 0
        right_end  = len(pretoken) - 1
        best_left_end    = +inf
        best_right_start = -inf

        left_tokens  = []
        right_tokens = []

        while left_start <= right_end:
            if best_left_end > right_end:  # Recompute
                for end in range(right_end, left_start-1, -1):
                    token = pretoken[left_start:end+1]
                    if token in self.vocab:
                        best_left_end = end
                        break
                else:
                    raise ValueError(f"Character not in vocab: {pretoken[left_start]}")

            if best_right_start < left_start:  # Recompute
                for start in range(left_start, right_end+1):
                    token = pretoken[start:right_end+1]
                    if token in self.vocab:
                        best_right_start = start
                        break
                else:
                    raise ValueError(f"Character not in vocab: {pretoken[right_end]}")

            # All the indices are
            left_score  = best_left_end - left_start
            right_score = right_end - best_right_start
            equal = left_score == right_score
            if left_score > right_score or (equal and self._prefer_left):
                left_tokens.append(pretoken[left_start:best_left_end+1])
                left_start = best_left_end+1
                best_left_end = +inf
            else:
                right_tokens.insert(0, pretoken[best_right_start:right_end+1])
                right_end = best_right_start-1
                best_right_start = -inf

        return left_tokens + right_tokens


class Xu(TokeniserWithVocabulary[WithSpecials]):
    """
    Models the inference method described in the VOLT paper (Xu 2021). It might just be an incorrect description of
    BPE, but it's worth actually attempting this weird algorithm:

    > After generating the vocabulary, VOLT uses a greedy strategy to encode text similar to BPE. To encode text,
    > it first splits sentences into characterlevel tokens. Then, we merge two consecutive tokens into one token if
    > the merged one is in the vocabulary. This process keeps running until no tokens can be merged. Out-of-vocabulary
    > tokens will be split into smaller tokens.

    So, it's a left-to-right approach, but only merges pairs. I also do a second iteration over the string.
    For example:

        a b c d e f g
        ab c d e f g
        abc d e f g
        abc de fg
        abcde fg

    Quite similar to left-to-right smallest-first.
    """

    def tokenise(self, word: str) -> Tokens:
        tokens = list(word)
        changed = True
        while changed:
            i = 0
            changed = False
            while i < len(tokens) - 1:
                merged = tokens[i] + tokens[i + 1]
                if merged in self.vocab:
                    tokens[i:i + 2] = [merged]
                    changed = True
                else:
                    i += 1
        return tokens
