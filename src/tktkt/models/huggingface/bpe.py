from typing import Optional, List, Tuple

from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import BPE

from .wrapper import HuggingFaceTokeniser
from ..bpe.vocabularisation import Merges
from ...interfaces import Preprocessor, Vocab


class HuggingFaceBPETokeniser(HuggingFaceTokeniser):
    """
    Constructs a BPE tokeniser with Rust backend.
    https://huggingface.co/docs/tokenizers/api/models#tokenizers.models.BPE
    """

    def __init__(self, vocab: Vocab, merges: Merges, dropout: float=0.0, preprocessor: Optional[Preprocessor]=None):
        backend_of_backend_of_backend = BPE(vocab=vocab, merges=merges, dropout=dropout if 0.0 < dropout <= 1.0 else None)
        backend_of_backend            = Tokenizer(model=backend_of_backend_of_backend)
        backend                       = PreTrainedTokenizerFast(tokenizer_object=backend_of_backend)

        super().__init__(backend)
        if preprocessor is not None:
            self.preprocessor = preprocessor
