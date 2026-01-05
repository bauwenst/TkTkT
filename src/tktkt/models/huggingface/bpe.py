from typing import Optional, Union

from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import BPE

from .wrapper import HuggingFaceTokeniser
from ...interfaces import Preprocessor, Vocab
from ...interfaces.identifiers import WithSpecials, AutoVocab


class HuggingFaceBPETokeniser(HuggingFaceTokeniser[WithSpecials]):
    """
    Constructs a BPE tokeniser with Rust backend.
    https://huggingface.co/docs/tokenizers/api/models#tokenizers.models.BPE
    """

    def __init__(self, vocab: Union[Vocab[WithSpecials], dict[str,int]], merges: list[tuple[str,str]],
                 dropout: float=0.0, preprocessor: Optional[Preprocessor]=None):
        backend_of_backend_of_backend = BPE(vocab=dict(vocab), merges=merges, dropout=dropout if 0.0 < dropout <= 1.0 else None)  # TODO: Known problem: because you don't use vocab.unsafe() here, HuggingFace gives you a "The OrderedVocab you are attempting to save contains holes for indices ..." print. This is harmless but annoying. https://github.com/huggingface/tokenizers/issues/1913
        backend_of_backend            = Tokenizer(model=backend_of_backend_of_backend)
        backend                       = PreTrainedTokenizerFast(tokenizer_object=backend_of_backend)
        super().__init__(backend, vocab_metadata=vocab)
        if preprocessor is not None:
            self.preprocessor = preprocessor

        self.vocab  = vocab
        self.merges = merges
        self._dropout_as_string = str(dropout)  # Saving it explicitly as a string because we do not want to give the impression you can change this.

    def getName(self) -> str:
        return super().getName() + "(p=" + self._dropout_as_string + ")"
