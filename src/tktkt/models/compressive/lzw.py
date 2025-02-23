"""
Tokenisation with the LZW compression algorithm, as mentioned in Zouhar et al. (2023)
    https://aclanthology.org/2023.acl-long.284v2.pdf
and first implemented in
    https://github.com/zouharvi/tokenization-principle
I've reimplemented the LZW vocabulariser to support TkTkT's preprocessors and corpora, and also made stuff faster and correcter.
"""
from pathlib import Path
from typing import Tuple

from ...interfaces.vocabulariser import Vocabulariser, UnidentifiedVocab, NamedIterable
from ...interfaces.tokeniser import Tokeniser, Preprocessor
from ...models.greedy.directional import L2R_Greedy
from ...util.printing import wprint
from ...util.iterables import streamProgress


class LzwVocabulariser(Vocabulariser):

    def __init__(self, preprocessor: Preprocessor, vocab_size: int):
        super().__init__(name="lzw", preprocessor=preprocessor)
        self.vocab_size = vocab_size

    @classmethod
    def _load(cls, file_or_folder: Path) -> UnidentifiedVocab:
        if file_or_folder.is_dir():
            file_or_folder = file_or_folder / "vocab.txt"

        with open(file_or_folder, "r", encoding="utf-8") as handle:
            for line in handle:
                yield line.rstrip()

    def _vocabulariseFromWords(self, word_iterable: NamedIterable[Tuple[str,int]]) -> Path:
        raise NotImplementedError

    def _vocabulariseFromSentences(self, sentence_iterable: NamedIterable[str]) -> Path:
        parent_folder = self._makeOutputFolder(extra_suffix=sentence_iterable.name)

        # Load the entire corpus into memory. Every sentence is split into pretokens, and each pretoken is split into characters.
        wprint(f"Loading and preprocessing '{sentence_iterable.name}' into memory...")
        tokens_of_pretokens = list(sentence_iterable.tqdm().flatmap(self.preprocessor.do).map(tuple))

        # Initialise with characters that should be recognised.
        alphabet = self.preprocessor.getAlphabet()
        if alphabet is None:
            alphabet = set()
            for pretoken in tokens_of_pretokens:
                for token in pretoken:
                    alphabet.update(token)
        else:
            alphabet = set(alphabet.getCharacters())

        marker = self.preprocessor.getBoundaryMarker()
        if marker is not None:
            alphabet.add(marker.substitute)

        # LZW works as follows:
        #   - Walk through the input aggregating chunks as large as are known to you. Each time you build a known chunk,
        #     remember the unknown chunk that is 1 longer than that and count it as valid next time you see it.
        #   - In the end, you have a set of candidate types and a subset of those that were actually used after they
        #     were remembered. The latter is the subword vocabulary.
        vocabulary_candidates = set(alphabet)  # <--- This is the actual Lempel-Ziv dictionary.
        vocabulary_found      = set(alphabet)  # <--- This is the subword vocabulary.
        ordered_vocabulary = sorted(alphabet)
        while len(vocabulary_found) < self.vocab_size:  # Note: This outer loop may actually iterate over the corpus more than once, which is unusual for LZW compression (because in LZW, you should only be grouping bits/characters, not LZW groups themselves). If this is unclear, this is on the same topic: https://cs.stackexchange.com/questions/9311/behavior-of-iterative-application-of-lz77
            # Iterate through the pretokens doing LZW compression.
            old_vocabulary_size = len(vocabulary_found)
            for idx_pretoken, tokens_of_pretoken_i in enumerate(streamProgress(tokens_of_pretokens, show_as="Compressing pretokens")):
                if len(tokens_of_pretoken_i) <= 1:
                    continue

                new_tokens = []
                idx_token = 0
                while idx_token < len(tokens_of_pretoken_i):
                    # Go through the pretoken's tokens from left to right and aggregate as many as you can at once.
                    n_tokens = 2
                    suggested_token = "".join(tokens_of_pretoken_i[idx_token:idx_token+n_tokens])
                    while suggested_token in vocabulary_candidates and idx_token+n_tokens <= len(tokens_of_pretoken_i):  # When the second condition triggers, token_found will be the same as suggested_token.
                        n_tokens += 1
                        suggested_token = "".join(tokens_of_pretoken_i[idx_token:idx_token+n_tokens])

                    known_token = "".join(tokens_of_pretoken_i[idx_token:idx_token+n_tokens-1])
                    new_tokens.append(known_token)
                    if known_token not in vocabulary_found:
                        vocabulary_found.add(known_token)
                        ordered_vocabulary.append(known_token)

                    vocabulary_candidates.add(suggested_token)
                    idx_token += n_tokens-1
                    if len(vocabulary_found) >= self.vocab_size:
                        break

                tokens_of_pretokens[idx_pretoken] = tuple(new_tokens)
                if len(vocabulary_found) >= self.vocab_size:
                    break

            wprint("Corpus fully iterated (and/or vocab limit reached).")
            wprint("\tAll suggested types:", len(vocabulary_candidates))
            wprint("\tAlready in the vocabulary:", len(vocabulary_found))
            wprint("\tCould be in the vocabulary:", len(vocabulary_candidates - vocabulary_found))

            if old_vocabulary_size == len(vocabulary_found):
                print("Early termination because vocabulary did not increase in size.")
                break

        # Write out
        vocab_path = parent_folder / "vocab.txt"
        with open(vocab_path, "w", encoding="utf-8") as handle:
            for typ in ordered_vocabulary:
                handle.write(typ + "\n")

        return vocab_path


class LzwTokeniser(L2R_Greedy):  # Lempel-Ziv is very explicitly a left-to-right algorithm.
    pass
