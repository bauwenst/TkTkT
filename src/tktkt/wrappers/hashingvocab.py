from typing import Mapping, List

from ..interfaces.tokeniser import Tokeniser, TokeniserWithVocab, Dict


class HashingMapping(Mapping):

    def __init__(self, hash_size: int):
        self.size = hash_size

    def get(self, key: str) -> int:
        return hash(key) % self.size  # TODO: I wonder which has function the CANINE paper uses. We should likely allow custom string hashing functions.

    def keys(self):
        # TODO: Should, at the very least, support special tokens. (TktktToHuggingFace needs to be able to check that
        #  the specials you give to the HF constructor have a well-defined ID, and we don't want to hash to
        #  special IDs.) Arguably, what we actually want is a Mapping that has one extra method "set(key, value)" that
        #  is linked to a dictionary (regardless of the type of mapping).
        #  |
        #  You would implement this in a parent that inherits from Mapping, and have subclasses implement a method that
        #  lets the parent check if a certain ID exists already.
        raise RuntimeError("Hashing mapping has an infinite key domain.")

    def values(self):
        return range(0, self.size)

    def items(self):
        raise RuntimeError("Hashing mapping has an infinite key domain.")


class TokeniserWithHashingVocab(TokeniserWithVocab):
    """
    Wraps an existing Tokeniser such that it gets a vocabulary mapping from tokens to IDs.

    The Tokeniser is allowed to produce an infinite amount of strings as tokens, which will be hashed to a finite set of
    integers and hence have an ID defined for all of them (sometimes overlapping).

    Note that this class has a fundamentally different usage than TokeniserWithVocabDict. The latter is used as the
    parent class for tokenisers that actually make use of a fixed set of types inside their algorithm, whilst this class
    is used as an afterthought to wrap tokenisers that don't use such a fixed set nor enforce that the produced tokens
    are inside one. Tokenisers that aren't defined as TokeniserWithVocabDict can't be wrapped into one, so if you still
    want a vocabulary for them, you should use this class (or possible future alternatives that are also dict-less).
    """

    def __init__(self, tokeniser: Tokeniser, vocab_size: int, default_id_decodes: Dict[int, str]=None):
        """
        :param default_id_decodes: An approximation of the reverse vocabulary mapping. An infinite amount of strings
                                   map to each ID, so an ID can technically not be decoded to one string.
                                   However, if you know the possible set of tokens for your application, you can enforce
                                   default decodings with a dictionary (although you should probably just be using a
                                   TokeniserWithVocabDict in that case).
        """
        super().__init__(tokeniser.preprocessor)
        self.core = tokeniser
        self.vocabulary_mapping = HashingMapping(vocab_size)
        self.default_decodings = default_id_decodes if default_id_decodes else dict()
        self._assertDefaultsMatch()

    def _assertDefaultsMatch(self):
        for id, typ in self.default_decodings.items():
            actual_id = self.typeToId(typ)
            if id != actual_id:
                raise ValueError(f"Default decoding of ID {id} is {typ}, yet this hashes to ID {actual_id}.")

    def tokenise(self, pretoken: str) -> List[str]:
        return self.core.tokenise(pretoken)

    def typeToId(self, t: str) -> int:
        return self.vocabulary_mapping.get(t)  # This cannot error nor give None (all strings have a vocab ID), so there is no need for UNK logic.

    def idToType(self, i: int) -> str:
        return self.default_decodings.get(i, f"[ID={i}]")

    def getVocabMapping(self) -> Mapping[str,int]:
        """
        Return an object which has the four methods .get(), .keys(), .values(), .items() representing the vocabulary.
        Does not have to be a dictionary!
        """
        return self.vocabulary_mapping

    def getVocabSize(self) -> int:
        return self.vocabulary_mapping.size
